# pyspice-mna

A SPICE netlist in, node voltages out — Modified Nodal Analysis assembled by
hand in numpy, no circuit-solver library involved. The whole simulator is one
file, [`mna.py`](mna.py), with its test suite in
[`test_mna.py`](test_mna.py).

```console
$ python mna.py divider.cir
* Voltage divider -- 10V across two 1k resistors
node          voltage (V)
1                      10
2                       5

source        current (A)
V1                 -0.005
```

## What it does

| Analysis | Directive | Notes |
|---|---|---|
| DC operating point | `.op` (or none) | the default when no sweep is requested |
| DC sweep | `.dc src start stop step` | steps one independent V or I source |
| AC / frequency response | `.ac dec points fstart fstop` | complex phasors, printed as dB and degrees |
| Transient | `.tran tstep tstop [be\|trap]` | fixed-step companion models, backward Euler or trapezoidal |
| Nonlinear DC | automatic with a diode | Newton-Raphson with voltage limiting |

Elements: resistors, independent voltage/current sources, capacitors,
inductors, all four controlled sources (VCVS/VCCS/CCVS/CCCS), and a Shockley
diode.

## Install

Requires Python 3.9+ and numpy (plus pytest to run the tests).

```bash
python -m venv .venv && source .venv/bin/activate && pip install numpy pytest
```

There is nothing to build or install — `mna.py` is a plain module.

## Quickstart

Write a netlist:

```
Voltage divider -- 10V across two 1k resistors
V1 1 0 10
R1 1 2 1k
R2 2 0 1k
.op
.end
```

and run it:

```bash
python mna.py divider.cir
```

Or drive it from Python:

```python
from mna import Circuit, Resistor, VoltageSource

c = Circuit()
c.add(VoltageSource("V1", "1", "0", 10.0))
c.add(Resistor("R1", "1", "2", 1000.0))
c.add(Resistor("R2", "2", "0", 1000.0))

voltages, currents = c.solve()   # {'1': 10.0, '2': 5.0}, {'V1': -0.005}
```

The same circuit objects feed the other analyses:

```python
from mna import ac_sweep, dc_sweep, solve_nonlinear_dc, transient

dc_sweep(c, "V1", 0.0, 10.0, 1.0)           # [(value, voltages, currents), ...]
ac_sweep(c, 10.0, 100e3, points_per_decade=10)   # complex phasors per frequency
transient(c, t_stop=5e-3, h=10e-6, method="trap")  # [(t, voltages, currents), ...]
solve_nonlinear_dc(c)                        # (voltages, currents, n_iterations)
```

## How it works

The unknown vector is `x = [node voltages..., branch currents...]` and the
solved system is `A x = z`, block-structured as

```
[ G   B ] [ v ]   [ i_sources ]
[ B^T D ] [ i ] = [ v_sources ]
```

`G` is the conductance matrix from resistors, and `B`/`B^T` couple in the
extra current unknown owned by each element that can't be written as a
conductance — voltage sources, VCVS, CCVS, and inductors. Ground (`0` or
`gnd`) is *eliminated* from the system rather than pinned to zero; that is
what keeps `A` nonsingular.

Every element only knows how to add its own contribution ("stamp") to `A` and
`z`, so the solver never special-cases element types. There are three stamps:
`stamp()` for DC into a real matrix, `stamp_ac()` for a complex matrix at
angular frequency ω, and `stamp_tran()` for one timestep of size `h`. Elements
that don't care about frequency or history inherit the DC stamp for the
other two.

Reactive elements in transient are replaced each step by their companion
model — a conductance plus a history-dependent source — and the result is
solved exactly like a DC operating point. Diodes work the same way, except
the companion is a tangent-line fit re-derived at every Newton iteration.

A few design choices worth calling out:

- An inductor is stamped as a branch current with `-jωL` on its own
  diagonal, not as a `1/(jωL)` admittance, so it degenerates cleanly to the
  DC short as ω → 0 instead of blowing up.
- Newton iteration on a diode uses a simplified version of SPICE's `pnjlim`
  voltage limiting. Without it the linear extrapolation overshoots deep into
  the exponential and oscillates forever; there's a test that shows exactly
  that.
- A node no element touches is reported by name as floating, rather than
  surfacing as a generic `LinAlgError` from numpy.

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
either on an element that can't use it is an error, not a silent no-op.

## Testing

```bash
pytest test_mna.py -q
```

91 tests. Results are checked against closed-form solutions rather than
spot-checked for plausibility:

- **AC** against the RC low-pass and RL high-pass transfer functions, including
  the −3 dB / −45° corner.
- **Transient** against the analytic RC and RL step responses, plus an
  energy-conservation check on a lossless LC tank that demonstrates backward
  Euler's numerical damping against trapezoidal's near-conservation.
- **The diode's** Newton-Raphson operating point against a closed form derived
  via the Lambert W function — computed by an independent Newton iteration
  written for the test, so it's a genuine cross-check rather than the solver
  grading its own homework.

Everything is verified through the Python API against hand-computed circuits
first, so the numerics are independent of the netlist format, and then again
end to end through the CLI.

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
