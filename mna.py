"""pyspice-mna -- a small SPICE-style circuit simulator in one file.

A netlist goes in, node voltages come out. The linear algebra is Modified
Nodal Analysis (MNA) assembled by hand into a dense numpy matrix; there is
no circuit-solver library underneath.

The unknown vector is ``x = [v_1, ..., v_n, i_1, ..., i_m]``: one row per
non-ground node, plus one branch-current row per element that can't be
written as a conductance (V, E, H, L). Ground ("0" or "gnd") is *removed*
from the system rather than pinned to zero, which is what keeps the matrix
nonsingular.

Each element only knows how to add its own contribution ("stamp") to the
system matrix ``A`` and right-hand side ``z``, given a node-name -> row
lookup ``idx`` and a branch-name -> row lookup ``vidx``. Three stamps exist:

  stamp()       DC operating point, real matrix. C is an open circuit, L a
                0V branch (a wire).
  stamp_ac()    small-signal AC at angular frequency omega, complex matrix.
                Frequency-independent elements just reuse stamp().
  stamp_tran()  one transient timestep of size h, using the companion model
                built from the previous step's history (v_prev, i_prev).
                History-independent elements just reuse stamp().

Sign conventions follow SPICE: for ``I n+ n- value`` the current flows from
n+ to n- *inside* the source; for ``V n+ n- value`` the extra unknown i_k is
the current from n+ to n- through the source. The dependent sources (E, G,
H, F) use the same convention.

Run it as a CLI:  python mna.py circuit.cir
"""

import argparse
import cmath
import math
import re
import sys
from dataclasses import dataclass, field

import numpy as np

GROUND_NAMES = {"0", "gnd"}


class CircuitError(ValueError):
    """Raised for any netlist, structural, or numerical problem."""


def _error(message, line_no=None, line_text=None):
    """Build a CircuitError, prefixed with netlist line info when known."""
    if line_no is not None:
        message = f"line {line_no}: {message}"
        if line_text:
            message += f" ({line_text.strip()!r})"
    return CircuitError(message)


# --------------------------------------------------------------------------
# Stamp primitives
#
# Every element stamp is built from these three moves. ``i``/``j`` are row
# indices, or None for ground (whose row was eliminated, so it is skipped).
# --------------------------------------------------------------------------


def _stamp_conductance(A, i, j, g):
    """Admittance g bridging nodes i and j (real or complex)."""
    if i is not None:
        A[i, i] += g
    if j is not None:
        A[j, j] += g
    if i is not None and j is not None:
        A[i, j] -= g
        A[j, i] -= g


def _stamp_current(z, i, j, value):
    """Current ``value`` flowing i -> j *through* the source."""
    if i is not None:
        z[i] -= value
    if j is not None:
        z[j] += value


def _stamp_branch(A, i, j, k):
    """Incidence terms tying branch-current row k to its terminals."""
    if i is not None:
        A[i, k] += 1.0
        A[k, i] += 1.0
    if j is not None:
        A[j, k] -= 1.0
        A[k, j] -= 1.0


# --------------------------------------------------------------------------
# Elements
# --------------------------------------------------------------------------


@dataclass
class Element:
    """Two-terminal base: name plus the nodes every element has.

    ``stamp_ac``/``stamp_tran`` default to the DC stamp, which is correct
    for every element whose behavior depends on neither frequency nor
    history (R, E, G, H, F, and independent sources in transient). C, L and
    the AC stimulus on V/I override them.
    """

    name: str
    n1: str
    n2: str

    def nodes(self):
        return (self.n1, self.n2)

    def stamp_ac(self, A, z, idx, vidx, omega):
        self.stamp(A, z, idx, vidx)

    def stamp_tran(self, A, z, idx, vidx, h, method, v_prev, i_prev):
        self.stamp(A, z, idx, vidx)


@dataclass
class Resistor(Element):
    value: float  # ohms

    def stamp(self, A, z, idx, vidx):
        _stamp_conductance(A, idx(self.n1), idx(self.n2), 1.0 / self.value)


@dataclass
class CurrentSource(Element):
    value: float  # amps, flowing n1 -> n2 through the source
    ac_mag: float = 0.0  # AC stimulus amplitude; 0 = inert in .ac
    ac_phase_deg: float = 0.0

    def stamp(self, A, z, idx, vidx):
        # In transient the source is held at this DC value for the whole run.
        _stamp_current(z, idx(self.n1), idx(self.n2), self.value)

    def stamp_ac(self, A, z, idx, vidx, omega):
        phasor = cmath.rect(self.ac_mag, math.radians(self.ac_phase_deg))
        _stamp_current(z, idx(self.n1), idx(self.n2), phasor)


@dataclass
class VoltageSource(Element):
    value: float  # volts, n1 - n2
    ac_mag: float = 0.0  # AC stimulus amplitude; 0 = inert in .ac
    ac_phase_deg: float = 0.0

    def stamp(self, A, z, idx, vidx):
        k = vidx(self.name)
        _stamp_branch(A, idx(self.n1), idx(self.n2), k)
        z[k] = self.value

    def stamp_ac(self, A, z, idx, vidx, omega):
        k = vidx(self.name)
        _stamp_branch(A, idx(self.n1), idx(self.n2), k)
        z[k] = cmath.rect(self.ac_mag, math.radians(self.ac_phase_deg))


@dataclass
class VCVS(Element):
    """E: v(n1) - v(n2) = gain * (v(ncp) - v(ncm)).

    Owns a branch-current unknown like a voltage source, but with RHS 0 --
    the constraint is a pure relation between node voltages, enforced by
    extra terms in the control columns of its own branch row.
    """

    ncp: str
    ncm: str
    gain: float

    def nodes(self):
        return (self.n1, self.n2, self.ncp, self.ncm)

    def stamp(self, A, z, idx, vidx):
        k = vidx(self.name)
        _stamp_branch(A, idx(self.n1), idx(self.n2), k)
        cp, cm = idx(self.ncp), idx(self.ncm)
        if cp is not None:
            A[k, cp] -= self.gain
        if cm is not None:
            A[k, cm] += self.gain


@dataclass
class VCCS(Element):
    """G: current gm * (v(ncp) - v(ncm)) flows n1 -> n2.

    No extra unknown -- a pure transconductance, stamped straight into the
    conductance block.
    """

    ncp: str
    ncm: str
    gm: float

    def nodes(self):
        return (self.n1, self.n2, self.ncp, self.ncm)

    def stamp(self, A, z, idx, vidx):
        i, j = idx(self.n1), idx(self.n2)
        cp, cm = idx(self.ncp), idx(self.ncm)
        for row, sign in ((i, 1.0), (j, -1.0)):
            if row is None:
                continue
            if cp is not None:
                A[row, cp] += sign * self.gm
            if cm is not None:
                A[row, cm] -= sign * self.gm


@dataclass
class CCVS(Element):
    """H: v(n1) - v(n2) = rm * i(vctrl).

    Owns a branch-current unknown and references another one. ``vctrl`` must
    name a V, E, or H element -- as in real SPICE, sensing the current
    through a resistor means inserting a 0V source in series with it.
    """

    vctrl: str
    rm: float

    def stamp(self, A, z, idx, vidx):
        k = vidx(self.name)
        _stamp_branch(A, idx(self.n1), idx(self.n2), k)
        A[k, vidx(self.vctrl)] -= self.rm


@dataclass
class CCCS(Element):
    """F: current beta * i(vctrl) flows n1 -> n2. No unknown of its own."""

    vctrl: str
    beta: float

    def stamp(self, A, z, idx, vidx):
        i, j = idx(self.n1), idx(self.n2)
        kc = vidx(self.vctrl)
        if i is not None:
            A[i, kc] += self.beta
        if j is not None:
            A[j, kc] -= self.beta


@dataclass
class Capacitor(Element):
    """C: open at DC, admittance j*omega*C in AC.

    In transient it becomes a Norton companion each step -- a conductance
    G_eq in parallel with a current source I_eq built from last step's
    terminal voltage (and, for trapezoidal, last step's current too):

        backward Euler:  G_eq = C/h,   I_eq = G_eq * v_prev
        trapezoidal:     G_eq = 2C/h,  I_eq = G_eq * v_prev + i_prev

    ``ic`` is the initial voltage v(n1)-v(n2) at t=0.
    """

    value: float  # farads
    ic: float = 0.0  # initial voltage, volts

    def stamp(self, A, z, idx, vidx):
        pass  # open circuit at DC steady state -- no contribution at all

    def stamp_ac(self, A, z, idx, vidx, omega):
        _stamp_conductance(A, idx(self.n1), idx(self.n2), 1j * omega * self.value)

    def stamp_tran(self, A, z, idx, vidx, h, method, v_prev, i_prev):
        g_eq = (1.0 if method == "be" else 2.0) * self.value / h
        i_eq = g_eq * v_prev + (0.0 if method == "be" else i_prev)
        i, j = idx(self.n1), idx(self.n2)
        _stamp_conductance(A, i, j, g_eq)
        _stamp_current(z, j, i, i_eq)  # j -> i: pushes current into n1

    def tran_current(self, h, method, v_now, v_prev, i_prev):
        """This step's current, from the companion model, for the history."""
        if method == "be":
            return self.value * (v_now - v_prev) / h
        return 2.0 * self.value / h * (v_now - v_prev) - i_prev


@dataclass
class Inductor(Element):
    """L: short (0V branch) at DC, impedance j*omega*L in AC.

    Stamping L as a branch current with -j*omega*L on its own diagonal --
    rather than as a 1/(j*omega*L) admittance -- keeps it well-conditioned
    as omega -> 0, where it degenerates cleanly to the DC short instead of
    blowing up.

    The same branch formulation absorbs the transient companion directly:
    the -j*omega*L diagonal becomes a real -R_eq and a history term V_eq
    lands on the RHS:

        backward Euler:  R_eq = L/h,   V_eq = R_eq * i_prev
        trapezoidal:     R_eq = 2L/h,  V_eq = R_eq * i_prev + v_prev

    ``ic`` is the initial current i_L at t=0.
    """

    value: float  # henries
    ic: float = 0.0  # initial current, amps

    def stamp(self, A, z, idx, vidx):
        k = vidx(self.name)
        _stamp_branch(A, idx(self.n1), idx(self.n2), k)
        z[k] = 0.0

    def stamp_ac(self, A, z, idx, vidx, omega):
        k = vidx(self.name)
        _stamp_branch(A, idx(self.n1), idx(self.n2), k)
        A[k, k] -= 1j * omega * self.value

    def stamp_tran(self, A, z, idx, vidx, h, method, v_prev, i_prev):
        k = vidx(self.name)
        _stamp_branch(A, idx(self.n1), idx(self.n2), k)
        r_eq = (1.0 if method == "be" else 2.0) * self.value / h
        v_eq = r_eq * i_prev + (0.0 if method == "be" else v_prev)
        A[k, k] -= r_eq
        z[k] = -v_eq


@dataclass
class Diode(Element):
    """D: nonlinear pn junction, I(v) = Is * (exp(v / (n*Vt)) - 1).

    Current flows n1 (anode) -> n2 (cathode) when forward biased, the same
    directional convention as every other two-terminal element here.

    Being nonlinear, a diode has no meaningful linear stamp: ``stamp()``
    raises rather than silently pretending to be an open circuit, and the
    inherited AC/transient stamps raise through it. The real stamp is
    ``stamp_linearized()``, the per-Newton-iteration companion used by
    ``solve_nonlinear_dc()``: a tangent-line fit at a guessed operating
    voltage v_op, i.e. a conductance g_eq = dI/dv(v_op) in parallel with a
    constant current correcting for the gap to the true I(v_op) --

        I(v) ~= I(v_op) + g_eq*(v - v_op) = g_eq*v + [I(v_op) - g_eq*v_op]

    which is the same "conductance + current correction" shape as the
    capacitor's transient companion, from a different physical equation.
    """

    Is: float = 1e-14  # saturation current, amps
    n: float = 1.0  # ideality factor
    Vt: float = 0.025852  # thermal voltage, volts (~300K)

    def current(self, v):
        return self.Is * (math.exp(v / (self.n * self.Vt)) - 1.0)

    def conductance(self, v):
        return (self.Is / (self.n * self.Vt)) * math.exp(v / (self.n * self.Vt))

    def stamp_linearized(self, A, z, idx, vidx, v_op):
        g_eq = self.conductance(v_op)
        i, j = idx(self.n1), idx(self.n2)
        _stamp_conductance(A, i, j, g_eq)
        _stamp_current(z, i, j, self.current(v_op) - g_eq * v_op)

    def stamp(self, A, z, idx, vidx):
        raise CircuitError(
            f"diode {self.name!r} is nonlinear -- only the DC operating point is "
            "supported: use solve_nonlinear_dc(circuit), not circuit.solve(), "
            "ac_sweep() or transient()"
        )


# Elements that own a branch-current unknown, i.e. that look like a voltage
# source to the MNA system (V, E, H, and L -- an inductor is a 0V branch at
# DC and gains a j*omega*L term on its own diagonal in AC).
_BRANCH_ELEMENTS = (VoltageSource, VCVS, CCVS, Inductor)


# --------------------------------------------------------------------------
# MNA assembly and solve
# --------------------------------------------------------------------------


class Circuit:
    """A netlist's elements plus the node/branch bookkeeping to solve them."""

    def __init__(self):
        self.elements = []
        self._node_index = {}  # node name -> row index (ground excluded)
        self._vsrc_index = {}  # branch name -> row index offset by n_nodes

    def add(self, element):
        for node in element.nodes():
            if not self._is_ground(node) and node not in self._node_index:
                self._node_index[node] = len(self._node_index)
        if isinstance(element, _BRANCH_ELEMENTS):
            if element.name in self._vsrc_index:
                raise CircuitError(f"duplicate branch element name {element.name!r}")
            self._vsrc_index[element.name] = len(self._vsrc_index)
        self.elements.append(element)
        return element

    @staticmethod
    def _is_ground(name):
        return str(name).lower() in GROUND_NAMES

    @property
    def n_nodes(self):
        return len(self._node_index)

    @property
    def n_vsrc(self):
        return len(self._vsrc_index)

    def _node_idx(self, name):
        return None if self._is_ground(name) else self._node_index[name]

    def _vsrc_idx(self, name):
        try:
            return self.n_nodes + self._vsrc_index[name]
        except KeyError:
            raise CircuitError(
                f"no voltage-source-style branch named {name!r} -- controlled "
                "sources (H/F) must reference a V, E, or H element already "
                "added to the circuit"
            ) from None

    def _assemble(self, dtype, stamp):
        """Build (A, z) by letting ``stamp(element, A, z)`` fill each row."""
        n = self.n_nodes + self.n_vsrc
        if n == 0:
            raise CircuitError("circuit has no nodes")
        A = np.zeros((n, n), dtype=dtype)
        z = np.zeros(n, dtype=dtype)
        for element in self.elements:
            stamp(element, A, z)
        return A, z

    def _solve_system(self, A, z):
        # A node whose row and column are entirely zero is touched by no
        # element at all; numpy would only report a generic LinAlgError,
        # which is useless for debugging a netlist.
        for name, i in self._node_index.items():
            if not np.any(A[i, :]) and not np.any(A[:, i]):
                raise CircuitError(f"node {name!r} is floating (not connected to any element)")
        try:
            x = np.linalg.solve(A, z)
        except np.linalg.LinAlgError as exc:
            raise CircuitError(
                "singular MNA matrix -- check for a loop of voltage sources "
                "or a node connected only to voltage/current sources with no "
                "resistive path to ground"
            ) from exc
        voltages = {name: x[i] for name, i in self._node_index.items()}
        currents = {name: x[self.n_nodes + i] for name, i in self._vsrc_index.items()}
        return voltages, currents

    def build(self):
        """Assemble and return the dense DC system (A, z), both real."""
        return self._assemble(float, lambda e, A, z: e.stamp(A, z, self._node_idx, self._vsrc_idx))

    def solve(self):
        """Solve the DC operating point.

        Returns (node_voltages, branch_currents), both dicts keyed by name.
        Ground is absent from node_voltages -- it is 0V by definition.
        """
        return self._solve_system(*self.build())

    def solve_ac(self, omega):
        """Solve the small-signal response at angular frequency omega (rad/s).

        Returns (node_voltages, branch_currents) as complex phasors. Only
        elements with a nonzero ``ac_mag`` drive the result; everything else
        is the linear network responding to that stimulus.
        """
        A, z = self._assemble(
            complex, lambda e, A, z: e.stamp_ac(A, z, self._node_idx, self._vsrc_idx, omega)
        )
        return self._solve_system(A, z)

    def solve_tran(self, h, method, state):
        """Solve one transient timestep of size h.

        ``state`` maps element name -> (v_prev, i_prev) and is consulted
        only by C and L. Returns real-valued results, like ``solve()``.
        """

        def stamp(e, A, z):
            v_prev, i_prev = state.get(e.name, (0.0, 0.0))
            e.stamp_tran(A, z, self._node_idx, self._vsrc_idx, h, method, v_prev, i_prev)

        return self._solve_system(*self._assemble(float, stamp))

    def solve_dc_newton(self, diode_voltages):
        """Solve one Newton iteration: diodes linearized at the given
        voltages (name -> v_op), everything else stamped normally."""

        def stamp(e, A, z):
            if isinstance(e, Diode):
                e.stamp_linearized(A, z, self._node_idx, self._vsrc_idx, diode_voltages[e.name])
            else:
                e.stamp(A, z, self._node_idx, self._vsrc_idx)

        return self._solve_system(*self._assemble(float, stamp))


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------


def dc_sweep(circuit, source_name, start, stop, step):
    """Step one independent V or I source and re-solve at each point.

    Returns a list of (sweep_value, node_voltages, branch_currents). The
    swept element's original value is restored afterwards either way.
    """
    element = next((e for e in circuit.elements if e.name == source_name), None)
    if element is None:
        raise CircuitError(f"'.dc' sweep source {source_name!r} not found in circuit")
    if not isinstance(element, (VoltageSource, CurrentSource)):
        raise CircuitError(
            f"'.dc' can only sweep an independent V or I source; "
            f"{source_name!r} is a {type(element).__name__}"
        )
    if step == 0:
        raise CircuitError("'.dc' step cannot be zero")
    n = round((stop - start) / step) + 1
    if n < 1:
        raise CircuitError("'.dc' sweep produces no points -- check start/stop/step signs")

    original = element.value
    results = []
    try:
        for value in np.linspace(start, stop, n):
            element.value = float(value)
            results.append((float(value), *circuit.solve()))
    finally:
        element.value = original
    return results


def ac_sweep(circuit, fstart, fstop, points_per_decade=10):
    """Log-spaced (per-decade) frequency sweep from fstart to fstop, in Hz.

    Returns a list of (freq_hz, node_voltages, branch_currents), the latter
    two holding complex phasors (see ``Circuit.solve_ac``).
    """
    if fstart <= 0 or fstop <= 0:
        raise CircuitError("'.ac' frequencies must be positive")
    if fstop < fstart:
        raise CircuitError("'.ac' fstop must be >= fstart")
    if points_per_decade < 1:
        raise CircuitError("'.ac' points-per-decade must be >= 1")

    if fstop == fstart:
        freqs = np.array([fstart])
    else:
        n = max(round(np.log10(fstop / fstart) * points_per_decade) + 1, 2)
        freqs = np.logspace(np.log10(fstart), np.log10(fstop), n)
    return [(float(f), *circuit.solve_ac(2.0 * np.pi * f)) for f in freqs]


def transient(circuit, t_stop, h, method="be"):
    """Fixed-timestep transient from t=0 to t_stop in steps of h.

    Each step replaces every C and L with its companion model built from
    the previous step's solution, leaving a purely resistive system solved
    exactly like a DC operating point -- the same technique real SPICE
    uses. Reactive elements start from their ``ic`` (0 by default), and
    independent sources are held at their DC value for the whole run, so a
    transient arises from an initial condition differing from the driven
    steady state rather than from a time-varying stimulus.

    The first returned point is t=h: the state at t=0 is by construction
    each element's ``ic``, so it is not separately re-solved.

    Returns a list of (t, node_voltages, branch_currents), real-valued.
    """
    if method not in ("be", "trap"):
        raise CircuitError(f"unknown transient method {method!r} (use 'be' or 'trap')")
    if h <= 0:
        raise CircuitError("'.tran' timestep must be positive")
    if t_stop <= 0:
        raise CircuitError("'.tran' stop time must be positive")

    reactive = [e for e in circuit.elements if isinstance(e, (Capacitor, Inductor))]
    state = {
        e.name: (e.ic, 0.0) if isinstance(e, Capacitor) else (0.0, e.ic) for e in reactive
    }

    results = []
    for step in range(1, round(t_stop / h) + 1):
        voltages, currents = circuit.solve_tran(h, method, state)
        new_state = {}
        for e in reactive:
            v_prev, i_prev = state[e.name]
            v_now = voltages.get(e.n1, 0.0) - voltages.get(e.n2, 0.0)
            if isinstance(e, Capacitor):
                new_state[e.name] = (v_now, e.tran_current(h, method, v_now, v_prev, i_prev))
            else:
                new_state[e.name] = (v_now, currents[e.name])
        state = new_state
        results.append((step * h, voltages, currents))
    return results


def solve_nonlinear_dc(circuit, max_iter=100, tol=1e-9, limit_voltage=True):
    """Solve the DC operating point of a circuit with diodes, via Newton-Raphson.

    Each iteration linearizes every diode about a guessed terminal voltage
    (see ``Diode.stamp_linearized``) and solves the resulting all-linear
    system; the guess is updated from the solution until the largest change
    drops below ``tol``. A circuit with no diodes converges in one
    iteration, giving exactly ``circuit.solve()``.

    Voltage limiting: a diode's I-V curve is a steep exponential and Newton
    extrapolates linearly from the current guess, so starting at v_op=0
    across even a modest source predicts a wildly optimistic forward
    voltage; the resulting huge conductance snaps the next guess back down
    and the iteration can oscillate forever. Real SPICE handles this with
    `pnjlim`; the limiter below is a simplified stand-in with the same idea
    -- cap how far the guess may move in one step once it is deep in the
    exponential region. Pass ``limit_voltage=False`` to see it diverge.

    Returns (node_voltages, branch_currents, n_iterations).
    """
    diodes = [e for e in circuit.elements if isinstance(e, Diode)]
    diode_voltages = {d.name: 0.0 for d in diodes}
    max_delta = 0.0

    for iteration in range(1, max_iter + 1):
        voltages, currents = circuit.solve_dc_newton(diode_voltages)

        max_delta = 0.0
        next_voltages = {}
        for d in diodes:
            v_old = diode_voltages[d.name]
            v_new = voltages.get(d.n1, 0.0) - voltages.get(d.n2, 0.0)
            if limit_voltage:
                v_crit = d.n * d.Vt * math.log(d.n * d.Vt / (math.sqrt(2) * d.Is))
                max_step = 4.0 * d.n * d.Vt
                if v_new > v_crit and (v_new - v_old) > max_step:
                    v_new = v_old + max_step
            max_delta = max(max_delta, abs(v_new - v_old))
            next_voltages[d.name] = v_new
        diode_voltages = next_voltages

        if max_delta < tol:
            return voltages, currents, iteration

    raise CircuitError(
        f"Newton-Raphson did not converge within {max_iter} iterations "
        f"(largest diode-voltage change was {max_delta:.3g}V, tol={tol:.3g}V)"
    )


# --------------------------------------------------------------------------
# Netlist parser
#
# Grammar (one statement per line):
#
#     <title line>          -- always the first line, ignored no matter what
#     * a comment line
#     Rname n+ n- value      ; inline comments after ';' are stripped too
#     Vname n+ n- dcvalue [AC mag [phase]]
#     Iname n+ n- dcvalue [AC mag [phase]]
#     Cname n+ n- value [IC=v0]
#     Lname n+ n- value [IC=i0]
#     Ename n+ n- nc+ nc- gain      ; VCVS
#     Gname n+ n- nc+ nc- gm        ; VCCS
#     Hname n+ n- vctrl rm          ; CCVS, vctrl names a V/E/H element
#     Fname n+ n- vctrl beta        ; CCCS, vctrl names a V/E/H element
#     Dname n+ n- [Is=.] [N=.] [Vt=.]   ; no .model cards, params inline
#     .op / .dc / .ac / .tran / .print / .end
#
# SPICE details this deliberately gets right, because they are the classic
# gotchas: the first line is ALWAYS a title and is discarded even if it
# looks like a valid element; a leading '+' continues the previous logical
# line; "meg" is 1e6 while "m" is 1e-3; trailing unit letters after a real
# prefix ("10kOhm") are ignored. Directives are parsed here but executed by
# the CLI, so a netlist naming an unsupported analysis still parses cleanly.
# --------------------------------------------------------------------------

_SI_SUFFIXES = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6,
    "m": 1e-3, "k": 1e3, "g": 1e9, "t": 1e12,
}
_VALUE_RE = re.compile(r"^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)([a-zA-Z]*)$")

# prefix -> (class, string fields before the trailing value, value's label).
# Construction is always cls(name, *fields, value), matching field order.
_ELEMENT_SPECS = {
    "R": (Resistor, ["n+", "n-"], "value"),
    "V": (VoltageSource, ["n+", "n-"], "value"),
    "I": (CurrentSource, ["n+", "n-"], "value"),
    "C": (Capacitor, ["n+", "n-"], "value"),
    "L": (Inductor, ["n+", "n-"], "value"),
    "E": (VCVS, ["n+", "n-", "nc+", "nc-"], "gain"),
    "G": (VCCS, ["n+", "n-", "nc+", "nc-"], "gm"),
    "H": (CCVS, ["n+", "n-", "vctrl"], "rm"),
    "F": (CCCS, ["n+", "n-", "vctrl"], "beta"),
}
_DIODE_PARAMS = {"IS": "Is", "N": "n", "VT": "Vt"}
_KNOWN_DIRECTIVES = {".op", ".end", ".dc", ".ac", ".tran", ".print"}


@dataclass
class Directive:
    name: str  # lowercased, with the leading '.', e.g. ".dc"
    args: list = field(default_factory=list)
    line_no: int = 0


@dataclass
class ParsedNetlist:
    circuit: Circuit
    directives: list
    title: str


def parse_value(token, line_no=None, line_text=None):
    """Parse a SPICE numeric literal with an optional engineering suffix.

    "1k" -> 1000.0, "4.7u" -> 4.7e-6, "1meg" -> 1e6 -- note that is NOT
    "1m" = 1e-3, the single most common netlist typo. Trailing unit letters
    after a recognized prefix ("10kOhm") are ignored.
    """
    match = _VALUE_RE.match(token)
    if not match:
        raise _error(f"invalid numeric value {token!r}", line_no, line_text)
    num_str, suffix = match.groups()
    suffix = suffix.lower()
    if suffix.startswith("meg"):
        mult = 1e6
    else:
        mult = _SI_SUFFIXES.get(suffix[:1], 1.0)
    return float(num_str) * mult


def _parse_extra_fields(cls, extra, usage, line_no, raw):
    """Parse an element line's optional trailing clause: 'AC mag [phase]'
    for V/I, or 'IC=<value>' for C/L. Returns constructor kwargs."""
    hint = ""
    if cls in (VoltageSource, CurrentSource):
        if extra[0].upper() == "AC" and len(extra) in (2, 3):
            return {
                "ac_mag": parse_value(extra[1], line_no, raw),
                "ac_phase_deg": parse_value(extra[2], line_no, raw) if len(extra) == 3 else 0.0,
            }
        if extra[0].upper() == "AC":
            hint = " (expected a trailing 'AC mag [phase]')"
    elif cls in (Capacitor, Inductor):
        if len(extra) == 1 and extra[0].upper().startswith("IC=") and len(extra[0]) > 3:
            return {"ic": parse_value(extra[0][3:], line_no, raw)}
        hint = " (expected a trailing 'IC=<value>')"
    raise _error(
        f"unexpected extra field(s) {' '.join(extra)!r} after '{usage}'{hint}", line_no, raw
    )


def _parse_diode(name, parts, line_no, raw):
    """D is the one element with all-optional KEY=value params, so it does
    not fit the single-trailing-value shape of _ELEMENT_SPECS."""
    if len(parts) < 3:
        raise _error(
            f"expected '{name} n+ n- [Is=..] [N=..] [Vt=..]' (at least 3 fields), "
            f"got {len(parts)}",
            line_no,
            raw,
        )
    kwargs = {}
    for token in parts[3:]:
        key, sep, val_tok = token.partition("=")
        attr = _DIODE_PARAMS.get(key.upper()) if sep else None
        if attr is None:
            raise _error(
                f"unknown diode parameter {token!r} (supported: Is=, N=, Vt=)", line_no, raw
            )
        if attr in kwargs:
            raise _error(f"diode parameter {key!r} given more than once", line_no, raw)
        kwargs[attr] = parse_value(val_tok, line_no, raw)
    return Diode(name, parts[1], parts[2], **kwargs)


def _join_continuations(raw_lines):
    """Merge lines starting with '+' into the previous logical line."""
    logical = []
    for physical_no, raw in enumerate(raw_lines, start=1):
        if raw.strip().startswith("+") and logical:
            prev_no, prev_text = logical[-1]
            logical[-1] = (prev_no, prev_text + " " + raw.strip()[1:].strip())
        else:
            logical.append((physical_no, raw))
    return logical


def parse_source(text):
    """Parse netlist text into a ParsedNetlist (circuit, directives, title)."""
    logical = _join_continuations(text.splitlines())
    if not logical:
        raise CircuitError("empty netlist (missing title line)")

    circuit = Circuit()
    directives = []

    for line_no, raw in logical[1:]:  # logical[0] is the title line
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("*"):
            continue

        parts = line.split()
        name = parts[0]

        if line.startswith("."):
            directive = name.lower()
            if directive == ".end":
                break
            if directive not in _KNOWN_DIRECTIVES:
                raise _error(f"unknown directive {name!r}", line_no, raw)
            directives.append(Directive(directive, parts[1:], line_no))
            continue

        prefix = name[0].upper()
        if prefix == "D":
            circuit.add(_parse_diode(name, parts, line_no, raw))
            continue

        spec = _ELEMENT_SPECS.get(prefix)
        if spec is None:
            raise _error(
                f"unknown element prefix {prefix!r} in {name!r} "
                "(supported: R, V, I, C, L, E, G, H, F, D)",
                line_no,
                raw,
            )
        cls, field_names, value_label = spec
        expected = len(field_names) + 2  # name + string fields + value
        usage = " ".join([name] + field_names + [value_label])
        if len(parts) < expected:
            raise _error(f"expected '{usage}' ({expected} fields), got {len(parts)}", line_no, raw)

        value = parse_value(parts[expected - 1], line_no, raw)
        extra = parts[expected:]
        kwargs = _parse_extra_fields(cls, extra, usage, line_no, raw) if extra else {}
        circuit.add(cls(name, *parts[1 : expected - 1], value, **kwargs))

    return ParsedNetlist(circuit, directives, logical[0][1].strip())


def parse_file(path):
    with open(path) as f:
        return parse_source(f.read())


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _node_sort_key(name):
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def _fmt(value):
    return f"{value:.6g}"


def _print_table(headers, rows):
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows), 10) + 2
        for col in range(len(headers))
    ]
    for row in [headers] + rows:
        print("".join(f"{v:<{widths[i]}}" for i, v in enumerate(row)))


def _sorted_nodes(results):
    return sorted(results[0][1], key=_node_sort_key)


def _run_dc(circuit, d):
    if len(d.args) != 4:
        raise _error(f"'.dc' expects 'source start stop step', got {len(d.args)}", d.line_no)
    source, *toks = d.args
    start, stop, step = (parse_value(t, d.line_no) for t in toks)
    results = dc_sweep(circuit, source, start, stop, step)

    nodes = _sorted_nodes(results)
    _print_table(
        [source] + nodes,
        [[_fmt(v)] + [_fmt(volts[n]) for n in nodes] for v, volts, _ in results],
    )


def _run_ac(circuit, d):
    if len(d.args) != 4:
        raise _error(f"'.ac' expects 'dec points fstart fstop', got {len(d.args)}", d.line_no)
    sweep_type, points_tok, fstart_tok, fstop_tok = d.args
    if sweep_type.lower() != "dec":
        raise _error(
            f"'.ac' sweep type {sweep_type!r} isn't supported (only 'dec' is implemented)",
            d.line_no,
        )
    results = ac_sweep(
        circuit,
        parse_value(fstart_tok, d.line_no),
        parse_value(fstop_tok, d.line_no),
        points_per_decade=int(parse_value(points_tok, d.line_no)),
    )

    nodes = _sorted_nodes(results)
    headers = ["freq_hz"]
    for n in nodes:
        headers += [f"V({n})_dB", f"V({n})_deg"]

    rows = []
    for freq, volts, _ in results:
        row = [_fmt(freq)]
        for n in nodes:
            mag = abs(volts[n])
            row += [
                _fmt(20.0 * np.log10(mag) if mag > 1e-300 else float("-inf")),
                _fmt(float(np.degrees(np.angle(volts[n])))),
            ]
        rows.append(row)
    _print_table(headers, rows)


def _run_tran(circuit, d):
    if len(d.args) not in (2, 3):
        raise _error(f"'.tran' expects 'tstep tstop [method]', got {len(d.args)}", d.line_no)
    tstep_tok, tstop_tok, *rest = d.args
    method = rest[0].lower() if rest else "trap"
    if method not in ("be", "trap"):
        raise _error(f"'.tran' method {method!r} isn't supported (use 'be' or 'trap')", d.line_no)
    results = transient(
        circuit,
        parse_value(tstop_tok, d.line_no),
        parse_value(tstep_tok, d.line_no),
        method=method,
    )

    nodes = _sorted_nodes(results)
    _print_table(
        ["t_s"] + nodes,
        [[_fmt(t)] + [_fmt(volts[n]) for n in nodes] for t, volts, _ in results],
    )


_SWEEP_RUNNERS = {".dc": _run_dc, ".ac": _run_ac, ".tran": _run_tran}


def _print_operating_point(circuit):
    if any(isinstance(e, Diode) for e in circuit.elements):
        voltages, currents, n_iter = solve_nonlinear_dc(circuit)
        print(f"note: converged via Newton-Raphson in {n_iter} iteration(s)", file=sys.stderr)
    else:
        voltages, currents = circuit.solve()

    print(f"{'node':<10}{'voltage (V)':>15}")
    for name in sorted(voltages, key=_node_sort_key):
        print(f"{name:<10}{_fmt(voltages[name]):>15}")
    if currents:
        print()
        print(f"{'source':<10}{'current (A)':>15}")
        for name in sorted(currents):
            print(f"{name:<10}{_fmt(currents[name]):>15}")


def main(argv=None):
    """Run the analyses a netlist asks for. Returns a process exit code.

    With no sweep directive this prints the DC operating point (via
    Newton-Raphson if the circuit contains diodes); each .dc/.ac/.tran
    directive prints one table.
    """
    parser = argparse.ArgumentParser(
        prog="mna", description="A SPICE-style circuit solver using Modified Nodal Analysis"
    )
    parser.add_argument("netlist", help="path to a .cir netlist file")
    args = parser.parse_args(argv)

    try:
        parsed = parse_file(args.netlist)
        print(f"* {parsed.title}")
        sweeps = [d for d in parsed.directives if d.name in _SWEEP_RUNNERS]
        for directive in sweeps:
            _SWEEP_RUNNERS[directive.name](parsed.circuit, directive)
            print()
        if not sweeps:
            _print_operating_point(parsed.circuit)
    except CircuitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
