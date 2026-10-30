"""Cross-validation against the R package {shorts}.

Every expected value below is printed in the {shorts} README (v3.2.0, same
sample data), so these tests check this package against the reference R
implementation without needing R.
"""

import pytest

import sprintfv as s


@pytest.fixture
def kimberley():
    return s.load_split_times("Kimberley")


def test_timing_gates(kimberley):
    f = s.fit_splits(kimberley["distance"], kimberley["time"])
    assert f.mss == pytest.approx(8.591, abs=1e-3)
    assert f.mac == pytest.approx(10.589, abs=1e-3)
    assert f.tau == pytest.approx(0.811, abs=1e-3)
    assert f.pmax_rel == pytest.approx(22.743, abs=1e-3)
    assert f.std_errors["mss"] == pytest.approx(0.123, abs=1e-3)


def test_timing_gates_fixed_time_correction(kimberley):
    f = s.fit_splits(kimberley["distance"], kimberley["time"] + 0.3)
    assert f.mss == pytest.approx(9.1278, abs=1e-3)
    assert f.mac == pytest.approx(6.6257, abs=1e-3)


def test_timing_gates_estimated_time_correction(kimberley):
    f = s.fit_splits(kimberley["distance"], kimberley["time"], correction="time")
    assert f.mss == pytest.approx(8.97, abs=0.01)
    assert f.mac == pytest.approx(7.27, abs=0.01)
    assert f.corrections["time_correction"] == pytest.approx(-0.235, abs=1e-3)


def test_radar():
    jim = s.load_radar_gun("Jim")
    f = s.fit_radar(jim["time"], jim["velocity"])
    assert f.mss == pytest.approx(7.99801, abs=1e-4)
    assert f.mac == pytest.approx(8.99871, abs=1e-4)
    assert f.corrections["time_correction"] == pytest.approx(-0.00011, abs=1e-5)
    assert f.std_errors["mac"] == pytest.approx(0.01997, abs=1e-4)


def test_radar_observed_mss():
    jim = s.load_radar_gun("Jim")
    f = s.fit_radar(jim["time"], jim["velocity"], use_observed_mss=True)
    assert f.mss == pytest.approx(8.095, abs=1e-3)
    assert f.mac == pytest.approx(8.67822, abs=1e-4)
    assert f.corrections["time_correction"] == pytest.approx(-0.01118, abs=1e-4)


def test_force_velocity_profile_analytic(kimberley):
    f = s.fit_splits(kimberley["distance"], kimberley["time"])
    fv = s.force_velocity_profile(f.mss, f.mac, body_mass=60, body_height=1.7, method="analytic")
    assert fv.f0 == pytest.approx(635, abs=0.5)
    assert fv.f0_rel == pytest.approx(10.6, abs=0.05)
    assert fv.v0 == pytest.approx(8.85, abs=0.005)
    assert fv.pmax == pytest.approx(1405, abs=0.5)
    assert fv.pmax_rel == pytest.approx(23.4, abs=0.05)
    assert fv.sfv_rel == pytest.approx(-1.2, abs=0.005)
