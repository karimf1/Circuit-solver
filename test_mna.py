"""Test suite for pyspice-mna: pytest test_mna.py

Two kinds of test here. Most build a circuit directly against the Python
API and check it against a hand-computed or closed-form answer, so the
numerics are verified independently of the text format. The last section
drives the CLI end to end on netlists written to a temp directory.
"""

import math

import numpy as np
import pytest

from mna import (
    CCCS,
    CCVS,
    VCCS,
    VCVS,
    Capacitor,
    Circuit,
    CircuitError,
    CurrentSource,
    Diode,
    Inductor,
    Resistor,
    VoltageSource,
    ac_sweep,
    dc_sweep,
    main,
    parse_source,
    parse_value,
    solve_nonlinear_dc,
    transient,
)

# ---------------------------------------------------------------------------
# Linear DC
# ---------------------------------------------------------------------------


def _divider(vs=10.0):
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", vs))
    c.add(Resistor("R1", "1", "2", 1000.0))
    c.add(Resistor("R2", "2", "0", 1000.0))
    return c


def test_voltage_divider():
    v, i = _divider().solve()

    assert v["1"] == pytest.approx(10.0)
    assert v["2"] == pytest.approx(5.0)
    # i_V1 is the current from n1 to n2 *through* the source, so a source
    # delivering power into the circuit shows a negative current.
    assert i["V1"] == pytest.approx(-0.005)


def test_current_source_into_resistor():
    # A current source pushes current out of n2 into the external circuit,
    # so wiring it 0 -> 1 injects 1mA into node 1.
    c = Circuit()
    c.add(CurrentSource("I1", "0", "1", 1e-3))
    c.add(Resistor("R1", "1", "0", 1000.0))

    v, _ = c.solve()

    assert v["1"] == pytest.approx(1.0)


def test_thevenin_equivalent_matches_direct_divider():
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 9.0))
    c.add(Resistor("R1", "1", "2", 2000.0))
    c.add(Resistor("R2", "2", "0", 4000.0))
    c.add(Resistor("Rload", "2", "0", 4000.0))  # in parallel with R2 -> 2k

    v, _ = c.solve()

    # V2 = 9 * (R2||Rload) / (R1 + R2||Rload) = 9 * 2000/4000
    assert v["2"] == pytest.approx(4.5)


def test_kcl_and_power_are_conserved():
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 12.0))
    c.add(Resistor("R1", "1", "2", 100.0))
    c.add(Resistor("R2", "2", "3", 200.0))
    c.add(Resistor("R3", "3", "0", 300.0))
    c.add(CurrentSource("I1", "0", "2", 0.01))

    v, i = c.solve()
    i_r1 = (v["1"] - v["2"]) / 100.0
    i_r2 = (v["2"] - v["3"]) / 200.0
    i_r3 = v["3"] / 300.0

    assert i_r1 + 0.01 == pytest.approx(i_r2)  # KCL at node 2
    assert i_r2 == pytest.approx(i_r3)  # KCL at node 3

    p_source = -v["1"] * i["V1"]
    p_resistors = i_r1**2 * 100.0 + i_r2**2 * 200.0 + i_r3**2 * 300.0
    p_current_source = 0.01 * v["2"]
    assert p_source + p_current_source == pytest.approx(p_resistors)


def test_ground_aliases_are_equivalent():
    c = Circuit()
    c.add(VoltageSource("V1", "1", "gnd", 3.3))
    c.add(Resistor("R1", "1", "0", 1000.0))

    v, _ = c.solve()

    assert v["1"] == pytest.approx(3.3)


def test_matrix_shape_matches_unknown_count():
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 5.0))
    c.add(VoltageSource("V2", "2", "0", 3.0))
    c.add(Resistor("R1", "1", "2", 1000.0))

    A, z = c.build()

    # 2 non-ground nodes + 2 voltage-source currents = 4x4.
    assert A.shape == (4, 4)
    assert z.shape == (4,)


def test_floating_node_raises_clear_error():
    # Node "2" gets registered by E1's control terminals but no element
    # actually conducts into it -- that must fail loudly, not as a cryptic
    # numpy singular-matrix error.
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 5.0))
    c.add(Resistor("R1", "1", "0", 100.0))
    c.add(VCCS("G1", "1", "0", "2", "0", 0.0))

    with pytest.raises(CircuitError, match="floating"):
        c.solve()


def test_voltage_source_loop_raises_clear_error():
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 5.0))
    c.add(VoltageSource("V2", "1", "0", 3.0))  # contradicts V1

    with pytest.raises(CircuitError, match="singular"):
        c.solve()


# ---------------------------------------------------------------------------
# Controlled sources (E / G / H / F)
# ---------------------------------------------------------------------------


def test_vcvs_forces_gain_times_control_voltage_and_draws_no_input_current():
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 2.0))
    c.add(VCVS("E1", "2", "0", "1", "0", 3.0))
    c.add(Resistor("R1", "2", "0", 1000.0))

    v, i = c.solve()

    assert v["2"] == pytest.approx(6.0)
    assert i["E1"] == pytest.approx(-v["2"] / 1000.0)
    assert i["V1"] == pytest.approx(0.0)  # ideal, infinite input impedance


def test_vccs_injects_transconductance_current():
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 1.0))
    c.add(VCCS("G1", "0", "2", "1", "0", 0.01))
    c.add(Resistor("R2", "2", "0", 1000.0))

    v, _ = c.solve()

    assert v["2"] == pytest.approx(0.01 * v["1"] * 1000.0)


def _current_sensed_circuit(controlled):
    # Vsense is a 0V "ammeter" in series with R1, purely to expose a branch
    # current -- the standard SPICE trick for current-controlled elements.
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 5.0))
    c.add(Resistor("R1", "1", "2", 1000.0))
    c.add(VoltageSource("Vsense", "2", "0", 0.0))
    c.add(controlled)
    c.add(Resistor("R2", "3", "0", 1000.0))
    return c


def test_ccvs_transresistance_from_sensed_current():
    c = _current_sensed_circuit(CCVS("H1", "3", "0", "Vsense", 500.0))

    v, i = c.solve()

    assert i["Vsense"] == pytest.approx(0.005)  # 5V / 1k through R1
    assert v["3"] == pytest.approx(500.0 * i["Vsense"])


def test_cccs_mirrors_sensed_current():
    c = _current_sensed_circuit(CCCS("F1", "0", "3", "Vsense", 2.0))

    v, i = c.solve()

    assert v["3"] == pytest.approx(2.0 * i["Vsense"] * 1000.0)


def test_missing_controlling_source_raises_clear_error():
    c = Circuit()
    c.add(VoltageSource("V1", "1", "0", 5.0))
    c.add(CCVS("H1", "2", "0", "Vghost", 500.0))
    c.add(Resistor("R1", "2", "0", 1000.0))

    with pytest.raises(CircuitError, match="Vghost"):
        c.solve()


def test_duplicate_branch_current_name_raises():
    c = Circuit()
    c.add(VoltageSource("X1", "1", "0", 5.0))

    with pytest.raises(CircuitError, match="duplicate"):
        c.add(VCVS("X1", "2", "0", "1", "0", 1.0))


# ---------------------------------------------------------------------------
# .dc sweep
# ---------------------------------------------------------------------------


def test_dc_sweep_hits_every_point_and_tracks_the_divider_ratio():
    results = dc_sweep(_divider(0.0), "V1", 0.0, 10.0, 2.0)

    assert [v for v, _, _ in results] == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    for value, voltages, _ in results:
        assert voltages["2"] == pytest.approx(value / 2.0)


def test_dc_sweep_restores_the_original_source_value():
    c = _divider(42.0)
    dc_sweep(c, "V1", 0.0, 10.0, 5.0)

    assert c.elements[0].value == pytest.approx(42.0)


@pytest.mark.parametrize(
    "args, match",
    [
        (("Vnope", 0.0, 10.0, 2.0), "not found"),
        (("R1", 100.0, 2000.0, 100.0), "R1"),  # not an independent source
        (("V1", 0.0, 10.0, 0.0), "step"),
    ],
)
def test_dc_sweep_rejects_bad_arguments(args, match):
    with pytest.raises(CircuitError, match=match):
        dc_sweep(_divider(), *args)


# ---------------------------------------------------------------------------
# Reactive elements: DC limits and AC analysis
# ---------------------------------------------------------------------------


def _rc_lowpass(r=1000.0, c=100e-9):
    circuit = Circuit()
    circuit.add(VoltageSource("V1", "1", "0", 5.0, ac_mag=1.0))
    circuit.add(Resistor("R1", "1", "2", r))
    circuit.add(Capacitor("C1", "2", "0", c))
    return circuit


def _rl_highpass(r=1000.0, ell=10e-3):
    circuit = Circuit()
    circuit.add(VoltageSource("V1", "1", "0", 5.0, ac_mag=1.0))
    circuit.add(Resistor("R1", "1", "2", r))
    circuit.add(Inductor("L1", "2", "0", ell))
    return circuit


def test_capacitor_is_open_and_inductor_is_short_at_dc():
    # No current flows through the cap at DC, so no drop across R1 and node
    # 2 sits at the source voltage; the inductor instead pulls it to ground.
    v, _ = _rc_lowpass().solve()
    assert v["2"] == pytest.approx(5.0)

    v, i = _rl_highpass().solve()
    assert v["2"] == pytest.approx(0.0)
    assert i["L1"] == pytest.approx(5.0 / 1000.0)


@pytest.mark.parametrize("freq_hz", [1.0, 159.15494309189535, 1e6])
def test_rc_lowpass_matches_analytic_transfer_function(freq_hz):
    r, c = 1000.0, 100e-9
    omega = 2 * np.pi * freq_hz

    voltages, _ = _rc_lowpass(r, c).solve_ac(omega)

    assert voltages["2"] == pytest.approx(1.0 / (1.0 + 1j * omega * r * c), rel=1e-9)


def test_rc_lowpass_corner_frequency_is_minus_3db_and_minus_45deg():
    r, c = 1000.0, 100e-9
    fc = 1.0 / (2 * np.pi * r * c)

    voltages, _ = _rc_lowpass(r, c).solve_ac(2 * np.pi * fc)
    v2 = voltages["2"]

    assert abs(v2) == pytest.approx(1.0 / np.sqrt(2.0))
    assert np.degrees(np.angle(v2)) == pytest.approx(-45.0)


@pytest.mark.parametrize("freq_hz", [0.0, 15915.494309189535, 1e6])
def test_rl_highpass_matches_analytic_transfer_function(freq_hz):
    # freq 0 is the point of stamping L as a branch current with -j*omega*L
    # on its diagonal: it degenerates to the DC short instead of blowing up
    # the way a 1/(j*omega*L) admittance would.
    r, ell = 1000.0, 10e-3
    omega = 2 * np.pi * freq_hz

    voltages, _ = _rl_highpass(r, ell).solve_ac(omega)
    analytic = (1j * omega * ell) / (r + 1j * omega * ell) if omega else 0.0

    assert voltages["2"] == pytest.approx(analytic, abs=1e-9)


def test_ac_sweep_spans_the_band_and_matches_analytic_at_every_point():
    r, c = 1000.0, 100e-9
    results = ac_sweep(_rc_lowpass(r, c), 10.0, 1000.0, points_per_decade=10)

    freqs = [f for f, _, _ in results]
    assert freqs[0] == pytest.approx(10.0)
    assert freqs[-1] == pytest.approx(1000.0)
    assert len(freqs) == 21  # 2 decades * 10 points/decade + 1

    for freq, voltages, _ in results:
        omega = 2 * np.pi * freq
        assert voltages["2"] == pytest.approx(1.0 / (1.0 + 1j * omega * r * c), rel=1e-9)


@pytest.mark.parametrize(
    "args, match", [((0.0, 1000.0), "positive"), ((1000.0, 10.0), "fstop")]
)
def test_ac_sweep_rejects_bad_frequency_range(args, match):
    with pytest.raises(CircuitError, match=match):
        ac_sweep(_rc_lowpass(), *args)


# ---------------------------------------------------------------------------
# Transient analysis
# ---------------------------------------------------------------------------


def _rc_charging(vs=5.0, r=1000.0, c=1e-6):
    circuit = Circuit()
    circuit.add(VoltageSource("V1", "1", "0", vs))
    circuit.add(Resistor("R1", "1", "2", r))
    circuit.add(Capacitor("C1", "2", "0", c, ic=0.0))
    return circuit, r * c


@pytest.mark.parametrize("method", ["be", "trap"])
def test_rc_charging_matches_analytic_step_response(method):
    vs = 5.0
    circuit, tau = _rc_charging(vs)

    results = transient(circuit, tau, tau / 1000.0, method=method)

    assert results[-1][1]["2"] == pytest.approx(vs * (1.0 - np.exp(-1.0)), rel=1e-3)


@pytest.mark.parametrize("method", ["be", "trap"])
def test_rl_charging_matches_analytic_step_response(method):
    vs, r, ell = 5.0, 1000.0, 10e-3
    circuit = Circuit()
    circuit.add(VoltageSource("V1", "1", "0", vs))
    circuit.add(Resistor("R1", "1", "2", r))
    circuit.add(Inductor("L1", "2", "0", ell, ic=0.0))
    tau = ell / r

    results = transient(circuit, tau, tau / 1000.0, method=method)

    assert results[-1][2]["L1"] == pytest.approx((vs / r) * (1.0 - np.exp(-1.0)), rel=1e-3)


@pytest.mark.parametrize(
    "method, expected_energy_ratio",
    [
        ("be", 0.0),  # backward Euler's numerical damping drains the tank
        ("trap", 1.0),  # trapezoidal conserves it
    ],
)
def test_lc_tank_shows_backward_eulers_numerical_damping(method, expected_energy_ratio):
    # A lossless LC tank with NO resistive path should ring forever. After
    # 20 periods, backward Euler has dissipated essentially all its energy
    # while trapezoidal still holds it -- the classic reason SPICE offers
    # both integrators.
    ell, c, ic_v = 1e-3, 1e-6, 1.0
    circuit = Circuit()
    circuit.add(Capacitor("C1", "1", "0", c, ic=ic_v))
    circuit.add(Inductor("L1", "1", "0", ell, ic=0.0))
    period = 2 * np.pi * np.sqrt(ell * c)

    _, voltages, currents = transient(circuit, 20 * period, period / 50.0, method=method)[-1]

    energy = 0.5 * c * voltages["1"] ** 2 + 0.5 * ell * currents["L1"] ** 2
    assert energy / (0.5 * c * ic_v**2) == pytest.approx(expected_energy_ratio, abs=0.05)


def test_transient_step_count_and_timestamps():
    circuit, tau = _rc_charging()
    h = tau / 10.0

    results = transient(circuit, tau, h, method="be")

    assert [t for t, _, _ in results] == pytest.approx([h * n for n in range(1, 11)])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "rk4"}, "method"),
        ({"h": 0.0}, "timestep"),
        ({"t_stop": 0.0}, "stop time"),
    ],
)
def test_transient_rejects_bad_arguments(kwargs, match):
    circuit, tau = _rc_charging()
    call = {"t_stop": tau, "h": tau / 10.0, "method": "be", **kwargs}

    with pytest.raises(CircuitError, match=match):
        transient(circuit, **call)


# ---------------------------------------------------------------------------
# Nonlinear DC: the diode and Newton-Raphson
# ---------------------------------------------------------------------------


def _lambertw(z):
    """Solve w*exp(w) = z for w (principal branch, z >= 0) by Newton's
    method. Deliberately an *independent* implementation, so validating the
    diode solver against it is a genuine cross-check, not a tautology (and
    it avoids a scipy dependency for one test)."""
    w = math.log(z) if z > math.e else z / math.e
    for _ in range(100):
        ew = math.exp(w)
        dw = (w * ew - z) / (ew * (w + 1))
        w -= dw
        if abs(dw) < 1e-14:
            break
    return w


def _diode_series_r(vs=5.0, r=1000.0):
    circuit = Circuit()
    circuit.add(VoltageSource("V1", "1", "0", vs))
    circuit.add(Resistor("R1", "1", "2", r))
    circuit.add(Diode("D1", "2", "0"))
    return circuit


@pytest.mark.parametrize("vs", [1.0, 3.0, 5.0, 9.0])
def test_diode_operating_point_matches_lambert_w_closed_form(vs):
    r, is_, n, vt = 1000.0, 1e-14, 1.0, 0.025852

    voltages, _, _ = solve_nonlinear_dc(_diode_series_r(vs, r))

    # Closed form of V = I*R + n*Vt*ln(I/Is), dropping the Shockley
    # equation's '-1' term (negligible here: I/Is ~ 1e11).
    z = is_ * r / (n * vt) * math.exp(vs / (n * vt))
    i_analytic = n * vt * _lambertw(z) / r
    assert (vs - voltages["2"]) / r == pytest.approx(i_analytic, rel=1e-8)


def test_diode_voltage_satisfies_shockley_equation_exactly():
    voltages, _, _ = solve_nonlinear_dc(_diode_series_r(vs=5.0))
    v_d = voltages["2"]

    i_through_r = (5.0 - v_d) / 1000.0
    i_shockley = 1e-14 * (math.exp(v_d / 0.025852) - 1.0)
    assert i_through_r == pytest.approx(i_shockley, rel=1e-8)


def test_reverse_biased_diode_blocks_current():
    circuit = Circuit()
    circuit.add(VoltageSource("V1", "1", "0", 5.0))
    circuit.add(Resistor("R1", "1", "2", 1000.0))
    circuit.add(Diode("D1", "0", "2"))  # anode at ground -> reverse biased

    voltages, _, _ = solve_nonlinear_dc(circuit)

    # Leakage is ~Is (1e-14 A), so node 2 floats up to essentially V1.
    assert voltages["2"] == pytest.approx(5.0, abs=1e-9)


def test_solve_nonlinear_dc_degenerates_to_a_linear_solve_without_diodes():
    circuit = _divider()

    voltages, currents, n_iter = solve_nonlinear_dc(circuit)
    linear_voltages, linear_currents = circuit.solve()

    assert n_iter == 1
    assert voltages == pytest.approx(linear_voltages)
    assert currents == pytest.approx(linear_currents)


def test_voltage_limiting_prevents_divergence_that_otherwise_occurs():
    voltages, _, n_iter = solve_nonlinear_dc(_diode_series_r(vs=10.0, r=100.0), max_iter=50)
    assert n_iter <= 50
    assert 0.0 < voltages["2"] < 1.0  # a sane forward drop, not garbage

    with pytest.raises(CircuitError, match="did not converge"):
        solve_nonlinear_dc(_diode_series_r(vs=10.0, r=100.0), max_iter=50, limit_voltage=False)


def test_diode_refuses_the_linear_analyses_with_a_clear_error():
    circuit = _diode_series_r()

    for call in (circuit.solve, lambda: circuit.solve_ac(1.0)):
        with pytest.raises(CircuitError, match="nonlinear"):
            call()


# ---------------------------------------------------------------------------
# Netlist parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token, expected",
    [
        ("1000", 1000.0),
        ("1k", 1000.0),
        ("4.7u", 4.7e-6),
        ("2.2p", 2.2e-12),
        ("1meg", 1e6),
        ("1MEG", 1e6),
        ("1m", 1e-3),  # the classic gotcha: 'm' is milli, not mega
        ("100meg", 100e6),
        ("10kOhm", 10e3),  # trailing unit letters after a real prefix
        ("5V", 5.0),  # unrecognized suffix -> unit annotation, ignored
        ("-3.3", -3.3),
        ("1.5e-6", 1.5e-6),
    ],
)
def test_parse_value_suffixes(token, expected):
    assert parse_value(token) == pytest.approx(expected)


def test_parse_value_rejects_garbage():
    with pytest.raises(CircuitError):
        parse_value("not_a_number")


def test_first_line_is_always_a_title_even_if_it_looks_like_an_element():
    parsed = parse_source("R1 1 0 100\nV1 1 0 5\n.end\n")

    assert parsed.title == "R1 1 0 100"
    assert [e.name for e in parsed.circuit.elements] == ["V1"]


def test_comments_continuations_and_dot_end():
    src = "title\n* comment\n\nV1 1\n+ 0\n+ 5\nR1 1 0 100 ; inline\n.end\nR2 1 0 1\n"
    parsed = parse_source(src)

    v1, r1 = parsed.circuit.elements
    assert (v1.n1, v1.n2, v1.value) == ("1", "0", 5.0)
    assert r1.value == pytest.approx(100.0)  # R2, after .end, was not parsed


def test_end_to_end_voltage_divider_matches_hand_solved_value():
    parsed = parse_source("divider\nV1 1 gnd 10\nR1 1 2 1k\nR2 2 0 1k\n.op\n.end\n")

    voltages, _ = parsed.circuit.solve()

    assert voltages["2"] == pytest.approx(5.0)


def test_directives_are_parsed_but_left_for_the_cli_to_execute():
    parsed = parse_source("title\nV1 1 0 5\nR1 1 0 100\n.tran 1u 1m\n.end\n")

    assert [(d.name, d.args) for d in parsed.directives] == [(".tran", ["1u", "1m"])]


@pytest.mark.parametrize(
    "src, match",
    [
        ("title\nQ1 1 0 5\n", "line 2"),
        ("title\nR1 1 0\n", r"expected 'R1 n\+ n- value'"),
        ("title\n.foo bar\n", "unknown directive"),
        ("title\nR1 1 0 100 AC 2\n", "unexpected extra field"),
        ("title\nV1 1 0 5 DC 2\n", "unexpected extra field"),
        ("title\nR1 1 0 100 IC=1\n", "unexpected extra field"),
        ("title\nC1 1 0 100n IC5\n", "unexpected extra field"),
        ("title\nD1 1\n", "expected"),
        ("title\nD1 1 0 Foo=1\n", "unknown diode parameter"),
        ("title\nD1 1 0 Is\n", "unknown diode parameter"),
        ("title\nD1 1 0 Is=1n Is=2n\n", "more than once"),
    ],
)
def test_malformed_lines_raise_located_errors(src, match):
    with pytest.raises(CircuitError, match=match):
        parse_source(src)


def test_ac_stimulus_clause_parses_magnitude_and_optional_phase():
    parsed = parse_source("title\nV1 1 0 5 AC 2 45\nI1 1 0 1 AC 3\nV2 2 0 5\n")
    v1, i1, v2 = parsed.circuit.elements

    assert (v1.value, v1.ac_mag, v1.ac_phase_deg) == (5.0, 2.0, 45.0)
    assert (i1.ac_mag, i1.ac_phase_deg) == (3.0, 0.0)
    assert v2.ac_mag == pytest.approx(0.0)  # no clause -> inert in .ac


def test_reactive_elements_and_their_ic_clause_parse():
    parsed = parse_source("title\nC1 1 2 100n IC=3.3\nL1 2 0 10m\n")
    c1, l1 = parsed.circuit.elements

    assert (c1.value, c1.ic) == (pytest.approx(100e-9), pytest.approx(3.3))
    assert (l1.value, l1.ic) == (pytest.approx(10e-3), pytest.approx(0.0))


def test_diode_parameters_default_and_parse_in_any_case_or_order():
    src = "title\nD1 1 0\nD2 1 0 vt=0.03 N=1.5 is=2n\n"
    default, explicit = parse_source(src).circuit.elements

    assert (default.Is, default.n, default.Vt) == (1e-14, 1.0, 0.025852)
    assert (explicit.Is, explicit.n, explicit.Vt) == (
        pytest.approx(2e-9),
        pytest.approx(1.5),
        pytest.approx(0.03),
    )


# ---------------------------------------------------------------------------
# Command line, end to end
# ---------------------------------------------------------------------------


def _run(tmp_path, capsys, source):
    netlist = tmp_path / "test.cir"
    netlist.write_text(source)
    rc = main([str(netlist)])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _rows(out):
    return [line.split() for line in out.splitlines() if line.strip()][1:]  # drop title


def test_cli_prints_the_operating_point(tmp_path, capsys):
    rc, out, _ = _run(tmp_path, capsys, "divider\nV1 1 0 10\nR1 1 2 1k\nR2 2 0 1k\n.op\n.end\n")

    assert rc == 0
    rows = _rows(out)
    assert ["2", "5"] in rows
    assert ["V1", "-0.005"] in rows


def test_cli_runs_a_dc_sweep_table(tmp_path, capsys):
    rc, out, _ = _run(
        tmp_path, capsys, "title\nV1 1 0 0\nR1 1 2 1k\nR2 2 0 1k\n.dc V1 0 10 2\n.end\n"
    )

    assert rc == 0
    rows = _rows(out)
    assert len(rows) == 1 + 6  # header + points at 0, 2, 4, 6, 8, 10
    assert rows[0][0] == "V1"
    assert rows[-1] == ["10", "10", "5"]


def test_cli_runs_an_ac_sweep_table(tmp_path, capsys):
    rc, out, _ = _run(
        tmp_path, capsys, "title\nV1 1 0 0 AC 1\nR1 1 2 1k\nC1 2 0 100n\n.ac dec 5 10 100k\n.end\n"
    )

    assert rc == 0
    assert _rows(out)[0] == ["freq_hz", "V(1)_dB", "V(1)_deg", "V(2)_dB", "V(2)_deg"]


def test_cli_runs_a_transient_table(tmp_path, capsys):
    rc, out, _ = _run(
        tmp_path, capsys, "title\nV1 1 0 5\nR1 1 2 1k\nC1 2 0 1u\n.tran 10u 5m\n.end\n"
    )

    assert rc == 0
    rows = _rows(out)
    assert rows[0][0] == "t_s"
    assert float(rows[-1][-1]) > 4.9  # RC = 1ms, run to 5*RC -> charged


def test_cli_uses_newton_raphson_when_the_circuit_has_a_diode(tmp_path, capsys):
    rc, out, err = _run(tmp_path, capsys, "title\nV1 1 0 5\nR1 1 2 1k\nD1 2 0\n.op\n.end\n")

    assert rc == 0
    assert "Newton-Raphson" in err
    v2 = next(float(row[1]) for row in _rows(out) if row[0] == "2")
    assert 0.0 < v2 < 1.0  # a sane forward drop


@pytest.mark.parametrize(
    "src, match",
    [
        ("title\nQ1 1 0 5\n", "line 2"),
        ("title\nV1 1 0 5\nR1 1 0 100\n.tran 1u 1m bogus\n.end\n", "'bogus'"),
        ("title\nV1 1 0 0 AC 1\nR1 1 0 1k\n.ac lin 10 1 1meg\n.end\n", "'lin'"),
        ("title\nV1 1 0 0\nR1 1 0 1k\n.dc Vnope 0 10 2\n.end\n", "not found"),
        ("title\nV1 1 0 0\nR1 1 2 1k\nD1 2 0\n.dc V1 0 5 1\n.end\n", "nonlinear"),
    ],
)
def test_cli_reports_errors_instead_of_tracebacks(tmp_path, capsys, src, match):
    rc, _, err = _run(tmp_path, capsys, src)

    assert rc == 1
    assert match in err


def test_cli_missing_file_reports_an_error(capsys):
    rc = main(["does_not_exist.cir"])

    assert rc == 1
    assert "error" in capsys.readouterr().err.lower()
