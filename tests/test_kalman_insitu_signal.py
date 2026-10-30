import numpy as np
import pytest

import sprintfv as s
from sprintfv import model
from sprintfv.kalman import KalmanParams, kalman_smooth, process_noise, transition_matrix


def _noisy_sine(dt=0.05):
    rng = np.random.default_rng(0)
    t = np.arange(0, 30, dt)
    return t, 5 * np.sin(t) + rng.normal(0, 0.05, t.size)


def test_kalman_matches_filterpy():
    filterpy = pytest.importorskip("filterpy")  # noqa: F841
    from filterpy.common import Q_discrete_white_noise
    from filterpy.kalman import KalmanFilter
    from scipy.linalg import expm

    dt = 0.05
    _, z = _noisy_sine(dt)
    kf = KalmanFilter(dim_x=4, dim_z=1)
    kf.P = np.diag(np.square([0.05, 0.1, 2, 5]))
    kf.Q = Q_discrete_white_noise(4, dt=dt, var=5)
    kf.R = 0.05**2
    kf.F = expm(np.diag(np.ones(3), 1) * dt)
    kf.H = np.array([[1.0, 0, 0, 0]])
    kf.x = np.array([[z[0]], [0], [0], [0]])
    mu, cov, _, _ = kf.batch_filter(z)
    xs, _, _, _ = kf.rts_smoother(mu, cov)
    assert np.allclose(kalman_smooth(z, dt, KalmanParams()), xs[:, :, 0], atol=1e-9)
    assert np.allclose(kalman_smooth(z, dt, KalmanParams(smoother=False)), mu[:, :, 0], atol=1e-9)


def test_kalman_matrices():
    assert np.allclose(transition_matrix(3, 0.1), [[1, 0.1, 0.005], [0, 1, 0.1], [0, 0, 1]])
    q = process_noise(2, 1.0, 1.0)
    assert np.allclose(q, [[1, 1], [1, 1]])


def test_kalman_estimates_derivatives():
    t, z = _noisy_sine()
    st = kalman_smooth(z, 0.05, KalmanParams())
    inner = slice(50, -50)
    assert np.abs(st[inner, 1] - 5 * np.cos(t[inner])).max() < 0.2
    assert np.abs(st[inner, 2] + 5 * np.sin(t[inner])).max() < 0.6


def test_kalman_parameters_change_smoothness():
    _, z = _noisy_sine()
    rough = kalman_smooth(z, 0.05, KalmanParams(process_var=500))[:, 2]
    smooth = kalman_smooth(z, 0.05, KalmanParams(process_var=0.5))[:, 2]
    assert np.std(np.diff(smooth)) < np.std(np.diff(rough))


def test_kalman_params_validation():
    with pytest.raises(ValueError):
        KalmanParams(order=1)
    with pytest.raises(ValueError):
        KalmanParams(order=3, initial_std=(1, 2))


def test_positions_to_kinematics_on_a_straight_sprint():
    dt = 0.05
    t = np.arange(0, 40, dt)
    x = model.distance_at_time(t, 8.0, 7.0) % 1000  # monotonic over 40 s (320 m)
    rng = np.random.default_rng(2)
    k = s.positions_to_kinematics(t, x + rng.normal(0, 0.03, t.size), dt=dt, min_segment=10)
    m = k.usable & (t > 3)
    assert np.abs(k.speed[m] - model.velocity_at_time(t[m], 8.0, 7.0)).max() < 0.15


def test_gaps_split_segments():
    dt = 0.05
    t = np.arange(0, 100, dt)
    x = np.sin(t)
    keep = (t < 40) | (t > 45)  # 5 s hole
    k = s.positions_to_kinematics(t[keep], x[keep], dt=dt, max_gap=1.0, min_segment=10)
    assert len(set(k.segment[k.segment >= 0])) == 2
    assert not k.usable[(k.time > 40) & (k.time < 45)].any()


@pytest.fixture(scope="module")
def lps_points():
    d = s.load_lps_session()
    k = s.positions_to_kinematics(d["time"], d["x"], d["y"], bounds=(-2, 32, -2, 17))
    return k.profile_points()


@pytest.mark.parametrize("method", s.IN_SITU_METHODS)
def test_in_situ_methods_on_real_lps(lps_points, method):
    p = s.fit_in_situ(*lps_points, method=method)
    assert 5 < p.a0 < 13 and 6 < p.s0 < 12
    assert p.peak_acceleration <= p.a0 + 1e-9
    assert p.n_points >= 20
    assert np.all(p.curve(np.linspace(0, p.s0, 50)) >= 0)


def test_hyperbolic_bends_at_low_speed(lps_points):
    p = s.fit_in_situ(*lps_points, method="hyperbolic")
    assert p.curve(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-9)
    assert p.peak_acceleration < p.a0


def test_linear_in_situ_recovers_a_line():
    rng = np.random.default_rng(3)
    v = rng.uniform(0, 9, 20000)
    a_max = 7.0 * (1 - v / 9.0)
    a = a_max * rng.uniform(0, 1, v.size) ** 4
    a[:400] = a_max[:400]  # a few maximal efforts
    p = s.fit_in_situ(v, a, method="linear", velocity_threshold=1.0)
    assert p.a0 == pytest.approx(7.0, rel=0.05)
    assert p.s0 == pytest.approx(9.0, rel=0.05)


def test_unknown_method():
    with pytest.raises(ValueError):
        s.fit_in_situ([1, 2, 3], [3, 2, 1], method="magic")


def test_clean_radar():
    rng = np.random.default_rng(4)
    t = np.arange(0, 6, 1 / 47)
    v = model.velocity_at_time(t, 9.0, 8.0) * 3.6 + rng.normal(0, 0.2, t.size)
    v[100] = 0.0           # dropped sample
    v[150] += 15           # spike
    c = s.clean_radar(t, v)
    assert c.unit == "kmh" and c.zeros_replaced == 1 and c.outliers_removed >= 1
    assert c.sampling_rate == pytest.approx(47, rel=1e-6)
    ok = np.isfinite(c.velocity)
    assert np.abs(c.velocity[ok][20:-20] - model.velocity_at_time(t[ok], 9.0, 8.0)[20:-20]).max() < 0.4
    f = s.fit_radar(c.time[ok], c.velocity[ok])
    assert f.mss == pytest.approx(9.0, rel=0.02)


def test_acceleration_end():
    t = np.arange(0, 8, 0.02)
    v = np.where(t < 5, model.velocity_at_time(t, 9.0, 8.0), model.velocity_at_time(5, 9.0, 8.0) - 2 * (t - 5))
    i = s.acceleration_end(t, v)
    assert 2.5 < t[i] < 5.0
