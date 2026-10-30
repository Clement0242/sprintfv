import numpy as np
import pytest

import sprintfv as s
from sprintfv import model


def test_time_and_distance_are_inverse():
    t = np.linspace(0, 6, 61)
    d = model.distance_at_time(t, 9.0, 8.0)
    # W is sqrt-like near its branch point, so very short distances lose a few digits.
    assert np.allclose(model.time_at_distance(d, 9.0, 8.0), t, atol=1e-6)
    assert model.time_at_distance(0.0, 9.0, 8.0) == pytest.approx(0.0, abs=1e-6)


def test_derivatives_are_consistent():
    t = np.linspace(0.01, 5, 500)
    v = model.velocity_at_time(t, 9.0, 8.0)
    inner = slice(1, -1)  # np.gradient is only first-order at the edges
    assert np.allclose(np.gradient(model.distance_at_time(t, 9.0, 8.0), t)[inner], v[inner], atol=1e-3)
    assert np.allclose(np.gradient(v, t)[inner], model.acceleration_at_time(t, 9.0, 8.0)[inner], atol=1e-2)
    assert model.acceleration_at_time(0.0, 9.0, 8.0) == pytest.approx(8.0)


@pytest.mark.parametrize("mss,mac", [(7.5, 6.0), (9.5, 9.0), (10.5, 11.0)])
def test_radar_recovers_known_parameters(mss, mac):
    rng = np.random.default_rng(1)
    t = np.arange(0, 6, 0.02)
    v = model.velocity_at_time(t - 0.05, mss, mac) + rng.normal(0, 0.05, t.size)
    f = s.fit_radar(t, v)
    assert f.mss == pytest.approx(mss, rel=0.005)
    assert f.mac == pytest.approx(mac, rel=0.02)
    assert f.corrections["time_correction"] == pytest.approx(0.05, abs=0.01)


def test_splits_with_flying_start():
    mss, mac, fd = 9.0, 8.0, 0.5
    d = np.array([5, 10, 15, 20, 30, 40.0])
    t = model.time_at_distance(d + fd, mss, mac) - model.time_at_distance(fd, mss, mac)
    f = s.fit_splits(d, t, correction="flying_distance")
    assert (f.mss, f.mac, f.corrections["flying_distance"]) == pytest.approx((mss, mac, fd), rel=1e-4)


def test_samozino_close_to_analytic():
    a = s.force_velocity_profile(9.0, 8.0, 75, method="analytic")
    b = s.force_velocity_profile(9.0, 8.0, 75)
    assert b.f0 == pytest.approx(a.f0, rel=0.02)
    assert b.v0 == pytest.approx(a.v0, rel=0.02)
    assert -15 < b.drf < 0 and 0 < b.rf_max < 100


def test_without_air_the_profile_is_the_exponential_model():
    fv = s.force_velocity_profile(9.0, 8.0, 75, body_height=1e-9, method="analytic")
    assert fv.f0_rel == pytest.approx(8.0, rel=1e-6)


def test_optimal_profile_simulation_matches_closed_form():
    from sprintfv.fvp import _time_to_distance
    # With no air resistance, a linear F-V profile *is* the exponential model.
    t_sim = _time_to_distance(8.0, 9.0, 30.0, 0.0)
    assert t_sim == pytest.approx(float(model.time_at_distance(30.0, 9.0, 8.0)), abs=1e-5)


def test_optimal_profile_is_a_minimum():
    fv = s.force_velocity_profile(9.0, 8.0, 75)
    opt = s.optimal_profile(fv, 30)
    assert opt.time_optimal <= opt.time_actual + 1e-9
    from sprintfv.fvp import _time_to_distance
    k = fv.k_air / fv.body_mass
    for sfv in (opt.sfv_opt * 0.8, opt.sfv_opt * 1.2):
        f0 = np.sqrt(-4 * fv.pmax_rel * sfv)
        assert _time_to_distance(f0, f0 / -sfv, 30, k) >= opt.time_optimal


def test_input_validation():
    with pytest.raises(ValueError):
        s.fit_splits([5, 10], [1.1, 1.9])
    with pytest.raises(ValueError):
        s.fit_splits([5, 10, 20], [1.1, 1.9, 3.1], correction="nope")
    with pytest.raises(ValueError):
        model.time_at_velocity(10.0, 9.0, 8.0)
