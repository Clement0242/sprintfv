"""Speed and acceleration from raw GPS / LPS positions: Kalman filter + RTS smoother.

Differentiating positions twice amplifies noise; a Kalman filter instead
estimates position, velocity, acceleration (and jerk) jointly, under a model
in which the highest derivative is driven by white noise. The
Rauch-Tung-Striebel (RTS) smoother then runs backwards, so every estimate uses
the samples before *and after* it, without the lag of a causal filter.

Every tuning parameter is exposed in :class:`KalmanParams`. The defaults are
the ones used for 20 Hz basketball LPS data (Kinexon) in the master's thesis
this package grew out of; for 10 Hz GPS, start by lowering ``process_var``.

Pure NumPy/SciPy: no ``filterpy`` dependency. With the same settings the
estimates match ``filterpy``'s ``batch_filter`` + ``rts_smoother`` (see the
tests), except that the initial covariance here is a proper diagonal matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal


@dataclass(frozen=True)
class KalmanParams:
    """Tuning of the Kalman filter.

    Attributes
    ----------
    order:
        Number of states: 2 = position, velocity; 3 = + acceleration;
        4 = + jerk (default). Acceleration needs at least 3.
    process_var:
        Variance of the white noise driving the highest derivative. Higher =
        the estimate follows the data more closely (and more noisily).
    measurement_std:
        Standard deviation of the position measurement noise (m).
    initial_std:
        Standard deviation of the initial state, one value per state.
        Defaults to (0.05, 0.1, 2, 5) truncated to ``order``.
    smoother:
        Run the RTS smoother after the forward pass (recommended offline).
    """

    order: int = 4
    process_var: float = 5.0
    measurement_std: float = 0.05
    initial_std: tuple[float, ...] | None = None
    smoother: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.order <= 6:
            raise ValueError("order must be between 2 and 6")
        if self.process_var <= 0 or self.measurement_std <= 0:
            raise ValueError("process_var and measurement_std must be positive")
        if self.initial_std is not None and len(self.initial_std) != self.order:
            raise ValueError(f"initial_std needs {self.order} values")

    def p0(self) -> np.ndarray:
        std = self.initial_std or (0.05, 0.1, 2.0, 5.0, 10.0, 20.0)[: self.order]
        return np.diag(np.square(std))


def transition_matrix(order: int, dt: float) -> np.ndarray:
    """F = expm(A·dt) for a chain of integrators (exact: A is nilpotent)."""
    F = np.zeros((order, order))
    for i in range(order):
        for j in range(i, order):
            F[i, j] = dt ** (j - i) / factorial(j - i)
    return F


def process_noise(order: int, dt: float, var: float) -> np.ndarray:
    """Discrete white-noise covariance, noise entering on the highest derivative.

    Q = var · G Gᵀ with G_i = dt^(n-1-i) / (n-1-i)!. For order 3 and 4 this is
    exactly ``filterpy.common.Q_discrete_white_noise``.
    """
    g = np.array([dt ** (order - 1 - i) / factorial(order - 1 - i) for i in range(order)])
    return var * np.outer(g, g)


def kalman_smooth(z: ArrayLike, dt: float, params: KalmanParams = KalmanParams()) -> np.ndarray:
    """Filter (and smooth) one coordinate sampled every ``dt`` seconds.

    Returns an array of shape (n, order): position, velocity, acceleration...
    NaN measurements are treated as missing (prediction only).
    """
    z = np.asarray(z, dtype=float).ravel()
    n, k = z.size, params.order
    F = transition_matrix(k, dt)
    Q = process_noise(k, dt, params.process_var)
    R = params.measurement_std**2

    x = np.zeros(k)
    first = np.flatnonzero(np.isfinite(z))
    if first.size == 0:
        raise ValueError("no finite measurement")
    x[0] = z[first[0]]
    P = params.p0()

    xs, Ps = np.empty((n, k)), np.empty((n, k, k))
    xps, Pps = np.empty((n, k)), np.empty((n, k, k))
    for i in range(n):
        # Predict, then update, for every sample including the first (as
        # filterpy's batch_filter): x0 / P0 describe the state one step before.
        x = F @ x
        P = F @ P @ F.T + Q
        xps[i], Pps[i] = x, P
        if np.isfinite(z[i]):  # update (H = [1, 0, ...])
            S = P[0, 0] + R
            K = P[:, 0] / S
            x = x + K * (z[i] - x[0])
            P = P - np.outer(K, P[0, :])
        xs[i], Ps[i] = x, P

    if not params.smoother:
        return xs
    for i in range(n - 2, -1, -1):  # RTS backward pass
        C = Ps[i] @ F.T @ np.linalg.inv(Pps[i + 1])
        xs[i] = xs[i] + C @ (xs[i + 1] - xps[i + 1])
        Ps[i] = Ps[i] + C @ (Ps[i + 1] - Pps[i + 1]) @ C.T
    return xs


@dataclass
class Kinematics:
    """Speed and acceleration estimated from positions, on a regular time grid.

    ``speed`` and ``acceleration`` are norms of the 2-D vectors (m/s, m/s²).
    ``accelerating`` is True when velocity and acceleration point the same
    way (v·a > 0): the samples an acceleration-speed profile is built from.
    ``usable`` excludes samples outside a valid segment and the trimmed edges.
    """

    time: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    tangential_acceleration: np.ndarray
    accelerating: np.ndarray
    usable: np.ndarray
    segment: np.ndarray = field(repr=False)

    def profile_points(self, max_speed: float = 11.5, max_acceleration: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
        """(speed, acceleration) of the usable accelerating samples, for :mod:`sprintfv.insitu`."""
        m = self.usable & self.accelerating & (self.speed < max_speed) & (self.acceleration < max_acceleration)
        return self.speed[m], self.acceleration[m]


def positions_to_kinematics(
    time: ArrayLike,
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    dt: float = 0.05,
    kalman: KalmanParams = KalmanParams(),
    prefilter_hz: float | None = 1.0,
    prefilter_order: int = 4,
    max_gap: float = 1.0,
    min_segment: float = 30.0,
    trim_fraction: float = 0.05,
    bounds: tuple[float, float, float, float] | None = None,
) -> Kinematics:
    """Full chain: resample, split into clean segments, (pre-filter,) Kalman + RTS.

    Parameters
    ----------
    time:
        Timestamps in seconds (any origin, irregular sampling accepted).
    x, y:
        Positions in metres. ``y`` may be omitted for straight-line data.
    dt:
        Step of the regular grid the data are resampled to (0.05 s = 20 Hz).
    kalman:
        Filter tuning, see :class:`KalmanParams`.
    prefilter_hz:
        Cut-off of a zero-phase Butterworth low-pass applied to positions
        before the Kalman filter (``None`` to skip it).
    max_gap:
        Longest gap (s) bridged by linear interpolation. Longer gaps split the
        recording into separate segments.
    min_segment:
        Segments shorter than this (s) are discarded.
    trim_fraction:
        Fraction of each segment, at both ends, marked as not usable: the filter
        has not converged at the start, and the smoother has no future at the end.
    bounds:
        (x_min, x_max, y_min, y_max): positions outside are treated as missing,
        e.g. the court plus a margin, to drop players on the bench.
    """
    t = np.asarray(time, dtype=float).ravel()
    coords = [np.asarray(x, dtype=float).ravel()]
    if y is not None:
        coords.append(np.asarray(y, dtype=float).ravel())
    if any(c.shape != t.shape for c in coords):
        raise ValueError("time and positions must have the same length")

    order = np.argsort(t, kind="stable")
    t = t[order] - t[order][0]
    coords = [c[order] for c in coords]
    _, uniq = np.unique(t, return_index=True)
    t, coords = t[uniq], [c[uniq] for c in coords]

    grid = np.arange(0.0, t[-1] + dt / 2, dt)
    cols = []
    for c in coords:
        ok = np.isfinite(c)
        ts = t[ok]
        g = np.interp(grid, ts, c[ok], left=np.nan, right=np.nan)
        # Do not bridge gaps longer than max_gap: a grid point is blanked when
        # the two samples around it are further apart than max_gap.
        nxt = np.clip(np.searchsorted(ts, grid), 0, ts.size - 1)
        prv = np.clip(nxt - 1, 0, ts.size - 1)
        exact = np.isclose(ts[nxt], grid)
        g[~exact & (ts[nxt] - ts[prv] > max_gap)] = np.nan
        cols.append(g)
    pos = np.column_stack(cols)

    valid = np.all(np.isfinite(pos), axis=1)
    if bounds is not None:
        x0, x1, y0, y1 = bounds
        valid &= (pos[:, 0] > x0) & (pos[:, 0] < x1)
        if pos.shape[1] > 1:
            valid &= (pos[:, 1] > y0) & (pos[:, 1] < y1)

    n = grid.size
    vel = np.full((n, pos.shape[1]), np.nan)
    acc = np.full((n, pos.shape[1]), np.nan)
    usable = np.zeros(n, dtype=bool)
    segment = np.full(n, -1)

    edges = np.diff(np.concatenate(([0], valid.astype(int), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    sos = (signal.butter(prefilter_order, prefilter_hz, "lp", fs=1.0 / dt, output="sos")
           if prefilter_hz else None)
    if kalman.order < 3:
        raise ValueError("acceleration needs a Kalman order of at least 3")

    for k, (s, e) in enumerate(zip(starts, ends)):
        if (e - s) * dt < min_segment:
            continue
        seg = pos[s:e]
        if sos is not None and e - s > 3 * (2 * prefilter_order + 1):
            seg = signal.sosfiltfilt(sos, seg, axis=0)
        for j in range(seg.shape[1]):
            states = kalman_smooth(seg[:, j], dt, kalman)
            vel[s:e, j], acc[s:e, j] = states[:, 1], states[:, 2]
        trim = int((e - s) * trim_fraction)
        usable[s + trim:e - trim] = True
        segment[s:e] = k

    speed = np.linalg.norm(vel, axis=1)
    acceleration = np.linalg.norm(acc, axis=1)
    dot = np.sum(vel * acc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tangential = np.where(speed > 0, dot / speed, 0.0)
    return Kinematics(
        time=grid, speed=speed, acceleration=acceleration,
        tangential_acceleration=tangential,
        accelerating=np.nan_to_num(dot) > 0, usable=usable & np.isfinite(speed),
        segment=segment,
    )
