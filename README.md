# pyspice-mna

A SPICE netlist in, node voltages out — Modified Nodal Analysis assembled by
hand in numpy, no circuit-solver library involved. The whole simulator is one
file, [`mna.py`](mna.py).

```
Voltage divider -- 10V across two 1k resistors        node    voltage (V)
V1 1 0 10                                       -->   1              10
R1 1 2 1k                                             2               5
R2 2 0 1k
.op                                                   source  current (A)
.end                                                  V1         -0.005
```

## Key features

| Analysis | Directive | Notes |
|---|---|---|
| DC operating point | `.op` (or none) | the default when no sweep is requested |
| DC sweep | `.dc src start stop step` | steps one independent V or I source |
| AC / frequency response | `.ac dec points fstart fstop` | complex phasors, printed as dB and degrees |
| Transient | `.tran tstep tstop [be\|trap]` | fixed-step companion models, backward Euler or trapezoidal |
| Nonlinear DC | automatic with a diode | Newton-Raphson with voltage limiting |

- **Elements**: resistors, independent voltage/current sources, capacitors,
  inductors, all four controlled sources (VCVS/VCCS/CCVS/CCCS), and a Shockley
  diode.
- **One extensible stamp interface.** Every element knows only how to add its
  own contribution to the matrix; the solver never branches on element type.
- **Two integration methods** for transient — backward Euler and trapezoidal —
  which makes numerical damping something you can see rather than read about.
- **A real netlist parser**, not a CSV reader: continuation lines, SPICE unit
  prefixes with the `meg`/`m` distinction, inline comments, arbitrary node names.
- **Usable as a library or as a CLI**, with the same objects driving every
  analysis.
- **Errors that name the thing that is wrong** — a floating node is reported by
  name, not as a singular-matrix traceback.

## How it works

The unknown vector is `x = [node voltages..., branch currents...]` and the
solved system is `A x = z`, block-structured as

```
[ G   B ] [ v ]   [ i_sources ]
[ B^T D ] [ i ] = [ v_sources ]
```

`G` is the conductance matrix from resistors, and `B`/`B^T` couple in the
extra current unknown owned by each element that can't be written as a
conductance — voltage sources, VCVS, CCVS, and inductors.

**Modified nodal analysis rather than plain nodal analysis** is the whole reason
for that second block. Plain nodal analysis cannot express an ideal voltage
source: there is no conductance that forces a node voltage. MNA's answer is to
add the branch current as an unknown and the constraint as an equation, which
costs one row and one column per such element and buys the ability to handle
ideal sources, inductors and current-controlled elements without approximating
any of them as a small resistance.

**Ground is eliminated from the system rather than pinned to zero.** Keeping a
ground row and setting it to `V = 0` works, but it leaves a redundant equation
and a matrix that is singular until you patch it. Deleting the row and column
makes `A` nonsingular by construction — the reference node is defined by its
absence.

**Every element only knows how to add its own contribution ("stamp")** to `A`
and `z`, so the solver never special-cases element types. There are three
stamps: `stamp()` for DC into a real matrix, `stamp_ac()` for a complex matrix
at angular frequency ω, and `stamp_tran()` for one timestep of size `h`.
Elements that don't care about frequency or history inherit the DC stamp for the
other two, so adding a new resistive element means writing one method.

**Reactive elements in transient are replaced each step by their companion
model** — a conductance plus a history-dependent source — and the result is
solved exactly like a DC operating point. That is the trick that makes one
linear solver serve all four analyses: differential equations become algebraic
ones, one timestep at a time. Diodes work the same way, except the companion is
a tangent-line fit re-derived at every Newton iteration.

A few design choices worth calling out:

- **An inductor is stamped as a branch current with `-jωL` on its own diagonal,
  not as a `1/(jωL)` admittance.** The admittance form goes to infinity as
  ω → 0; the branch-current form degenerates cleanly to the DC short. That
  matters because `.ac` sweeps routinely start near DC, and a formulation that
  blows up at the first frequency point is not usable.
- **Newton iteration on a diode uses a simplified version of SPICE's `pnjlim`
  voltage limiting.** Without it the linear extrapolation overshoots deep into
  the exponential and oscillates forever — the diode's `exp(V/nVt)` means a
  Newton step of a few hundred millivolts is a factor of e^10 in current.
  Limiting the per-iteration voltage change is what makes the nonlinear solve
  converge at all.
- **A node no element touches is reported by name as floating**, rather than
  surfacing as a generic `LinAlgError` from numpy. The information needed to say
  which node is in the data structure; not using it is a choice, and the wrong
  one.

## Netlist format

A small SPICE subset, one statement per line:

```
<title line>                       first line, always discarded
* full-line comment                ; inline comments are stripped too
Rname n+ n- value
Vname n+ n- dcvalue [AC mag [phase]]
Iname n+ n- dcvalue [AC mag [phase]]
Cname n+ n- value [IC=v0]
Lname n+ n- value [IC=i0]
Ename n+ n- nc+ nc- gain           VCVS
Gname n+ n- nc+ nc- gm             VCCS
Hname n+ n- vctrl rm               CCVS -- vctrl names a V/E/H element
Fname n+ n- vctrl beta             CCCS -- vctrl names a V/E/H element
Dname n+ n- [Is=..] [N=..] [Vt=..] diode -- no .model cards, params inline
.op / .dc / .ac / .tran / .end
```

The parsing details it deliberately gets right are the ones that separate
"reads a text file" from "reads a netlist":

- The **first line is always a title** and is discarded even when it looks
  like a perfectly valid element. Real decks always have one, and forgetting
  it silently eats your first component.
- A line starting with `+` **continues** the previous logical line.
- `meg` is 1e6 and `m` is 1e-3 — the single most common netlist typo. Unit
  letters trailing a real prefix (`10kOhm`, `100mA`) are ignored, as in SPICE.
- Node names are arbitrary case-sensitive strings; `0` and `gnd` both mean
  ground.
- `H`/`F` sense current through a named voltage-source branch, so measuring
  the current in a resistor means inserting a 0V source in series with it as
  an ammeter — again, exactly like SPICE.

`AC mag [phase]` only matters to `.ac`; `IC=` only matters to `.tran`. Putting
either on an element that can't use it is an error, not a silent no-op —
silently ignoring a directive the user meant is worse than refusing it.

## Limitations

- No `.subckt` / hierarchical netlists, no `.model` cards (diode parameters
  are inlined on the element line)
- No BJT or MOSFET models
- Dense `numpy.linalg.solve` only — fine at these sizes, not built for large
  sparse networks
- Fixed timestep in `.tran`, no adaptive step-size control
- No waveform sources (PULSE/SIN/PWL): a transient arises from a reactive
  element's initial condition differing from the driven steady state, not
  from a time-varying stimulus
- Diodes participate only in the DC operating point — nonlinear `.ac`/`.tran`
  around a Newton-solved bias point isn't implemented

## Possible improvements

- **Waveform sources (PULSE, SIN, PWL).** The single biggest gap in usefulness.
  Without them a transient can only be started from an initial condition, which
  rules out the most common thing anyone wants to simulate: a circuit's response
  to a driven input. The stamp interface already takes a timestep, so the source
  value just becomes a function of `t`.
- **Nonlinear transient and AC.** Once a driven stimulus exists, the natural
  next step is a Newton solve *inside* each timestep, and small-signal `.ac`
  linearised about a Newton-solved DC bias point. That combination is what turns
  the diode from a DC curiosity into a rectifier you can actually simulate, and
  it is the prerequisite for any active device.
- **A MOSFET model.** Level-1 Shichman-Hodges is about a page of code and is the
  gateway to every interesting circuit — amplifiers, switches, logic gates. It
  needs `.model` cards, which is the same parsing work as the next item.
- **`.model` cards and `.subckt` hierarchy.** Inline parameters do not scale past
  one or two devices. Both are parser features rather than solver features, so
  they can land without touching the numerics.
- **Adaptive timestep with local truncation error control.** Fixed-step
  transient forces a choice between accuracy and runtime that SPICE stopped
  making in 1973. LTE estimation from the difference between two integration
  orders is the standard route, and the trapezoidal/backward-Euler pair is
  already there to build it from.
- **Sparse matrices with an ordered factorisation.** Dense LU is `O(n³)`; real
  netlists are overwhelmingly sparse. `scipy.sparse` plus a fill-reducing
  ordering would make hundred-node circuits practical.
- **Gmin stepping and source stepping for nonlinear convergence.** `pnjlim`
  handles the well-behaved cases; the standard fallbacks handle the rest, and
  right now a non-converging circuit just fails.
- **Noise and sensitivity analyses.** Both reuse the AC machinery — noise is a
  weighted sum over element contributions at each frequency, and sensitivity
  falls out of the adjoint system — so they are cheap additions relative to what
  they add.
- **Plotting.** Every analysis returns lists of numbers that someone then has to
  plot themselves. A thin matplotlib wrapper would make the transient and AC
  results readable without leaving the tool.
