"""In-situ acceleration-speed profiles from training or match tracking data.

No sprint test is needed: the profile is the *upper envelope* of the
(speed, acceleration) cloud of everything the athlete did. Four methods:

``"linear"``
    Morin et al. (2021), as in {shorts}: the 2 highest accelerations per
    0.2 m/s band, above the speed of the single highest acceleration, then a
    straight line. Intercept = A0, x-intercept = S0.
``"hyperbolic"``
    The robust envelope method of F. Fenaux (HypExp), used in the master's
    thesis this package comes from. Inliers are chosen by density
    (k-nearest-neighbours) above the line a = 5 - v; then a curve is fitted by
    minimising  lambda * area under the curve + number of points above it,
    first a linear-exponential curve (LinExp), then the hyperbolic-exponential
    one (HypExp) started from it. Unlike a straight line it bends at low
    speed, where nobody produces their maximal acceleration in a game.
``"tukey"``
    Per 0.1 m/s band, outliers removed with Tukey's fences (local IQR), the
    maximum of what remains kept, then a straight line.
``"prediction_interval"``
    The 2 highest points per 0.1 m/s band, then points outside the 95 %
    prediction interval of the line are removed iteratively.

In the thesis (22 elite youth basketball players, LPS at 20 Hz, compared with
a 28 m sprint), A0 from match data was overestimated by 18-27 % relative to
the sprint test, while S0 was close for the linear-type methods: an in-situ
profile complements a sprint test, it does not replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.spatial import cKDTree

METHODS = ("linear", "hyperbolic", "tukey", "prediction_interval")
_BAD = 1e12  # objective value for parameters outside the physiological domain


@dataclass
class InSituProfile:
    """Result of an in-situ profile.

    Attributes
    ----------
    method:
        One of :data:`METHODS`.
    a0:
        Acceleration extrapolated to zero speed (m/s²), the usual "A0".
    s0:
        Speed at which acceleration reaches zero (m/s), the usual "S0".
    peak_acceleration:
        Highest acceleration on the fitted curve. Equal to ``a0`` for the
        straight-line methods; lower for the hyperbolic curve, which bends
        down at low speed.
    params:
        Raw parameters of the fitted curve.
    r2:
        R² of the line on the retained points (linear-type methods only).
    speed, acceleration:
        The points the curve was fitted on.
    """

    method: str
    a0: float
    s0: float
    peak_acceleration: float
    params: dict[str, float]
    curve: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    r2: float | None = None
    speed: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    acceleration: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def n_points(self) -> int:
        return int(self.speed.size)

    def summary(self) -> str:
        r2 = f"   R² {self.r2:.3f}" if self.r2 is not None else ""
        return (f"In-situ profile ({self.method}): A0 {self.a0:.2f} m/s², S0 {self.s0:.2f} m/s, "
                f"peak {self.peak_acceleration:.2f} m/s², {self.n_points} points{r2}")


# ---------------------------------------------------------------- helpers

def _points(velocity: ArrayLike, acceleration: ArrayLike, max_speed: float, max_acceleration: float) -> np.ndarray:
    v = np.asarray(velocity, dtype=float).ravel()
    a = np.asarray(acceleration, dtype=float).ravel()
    if v.shape != a.shape:
        raise ValueError("velocity and acceleration must have the same length")
    keep = np.isfinite(v) & np.isfinite(a) & (v > 0) & (a > 0) & (v < max_speed) & (a < max_acceleration)
    return np.column_stack((v[keep], a[keep]))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def _line_profile(method: str, v: np.ndarray, a: np.ndarray, extra: dict | None = None) -> InSituProfile:
    if v.size < 3:
        raise ValueError(f"{method}: fewer than 3 points retained, cannot fit a line")
    slope, intercept = np.polyfit(v, a, 1)
    if slope >= 0:
        raise ValueError(f"{method}: acceleration does not decrease with speed (no maximal efforts?)")
    a0, s0 = float(intercept), float(-intercept / slope)
    return InSituProfile(
        method=method, a0=a0, s0=s0, peak_acceleration=a0,
        params={"slope": float(slope), "intercept": a0, **(extra or {})},
        curve=lambda x: np.clip(intercept + slope * np.asarray(x, dtype=float), 0, None),
        r2=_r2(a, intercept + slope * v), speed=v, acceleration=a,
    )


def density_inliers(
    pts: np.ndarray,
    *,
    k_neighbors: int = 50,
    density_quantile: float = 0.99,
    floor_line: tuple[float, float] = (-1.0, 5.0),
) -> np.ndarray:
    """Inlier mask used by the hyperbolic, Tukey and interval methods.

    A point is kept when it lies above the line a = floor_line[0]·v +
    floor_line[1] (walking and jogging are not acceleration efforts) and in a
    dense region: its distance to its ~k/2-th neighbour is below the
    ``density_quantile`` of the distances to the k-th neighbour (isolated
    points are tracking artefacts more often than real efforts).
    """
    k = min(k_neighbors, len(pts))
    dist, _ = cKDTree(pts).query(pts, k=k)
    dist = dist.reshape(len(pts), -1)
    q = np.quantile(dist[:, -1], density_quantile)
    dense = dist[:, min(k_neighbors // 2 - 1, k - 1)] <= q
    above = pts[:, 1] > np.polyval(floor_line, pts[:, 0])
    return dense & above


# ---------------------------------------------------------------- methods

def _linear(pts: np.ndarray, velocity_step: float = 0.2, n_per_bin: int = 2,
            velocity_threshold: float | None = None) -> InSituProfile:
    v, a = pts[:, 0], pts[:, 1]
    # Right-closed bands (lo, hi], first one closed, as R's cut() in {shorts}.
    bins = np.maximum(np.ceil(v / velocity_step).astype(int) - 1, 0)
    order = np.lexsort((-a, bins))
    sel = np.concatenate([order[bins[order] == b][:n_per_bin] for b in np.unique(bins)])
    vs, as_ = v[sel], a[sel]
    if velocity_threshold is None:
        velocity_threshold = float(vs[np.argmax(as_)])
    keep = vs >= velocity_threshold
    return _line_profile("linear", vs[keep], as_[keep], {"velocity_threshold": velocity_threshold})


def linexp(v: ArrayLike, a0: float, vm: float, v1: float) -> np.ndarray:
    """LinExp curve: a(v) = (a0 - a0/vm · v) · (1 - exp(-v/v1))."""
    v = np.asarray(v, dtype=float)
    with np.errstate(all="ignore"):
        return (a0 - a0 / vm * v) * (1 - np.exp(-v / v1))


def _linexp_area(a0: float, vm: float, v1: float) -> float:
    """Exact integral of :func:`linexp` over [0, vm]."""
    k, lam = a0 / vm, 1 / v1
    with np.errstate(all="ignore"):
        at_vm = -k * vm**2 / 2 + a0 * vm + (a0 / lam - k / lam**2 * (lam * vm + 1)) * np.exp(-lam * vm)
        return at_vm - (a0 / lam - k / lam**2)


def hypexp(v: ArrayLike, vm: float, a0: float, v0: float, v1: float) -> np.ndarray:
    """HypExp curve: a(v) = sqrt(a0/v0 · (2·v0 + vm - v) · (vm - v)) · sqrt(1 - exp(-v/v1))."""
    v = np.asarray(v, dtype=float)
    vr = vm - v
    y2 = (a0 / v0) * (2 * v0 + vr) * vr
    with np.errstate(all="ignore"):
        return np.sqrt(np.abs(y2)) * np.sqrt(np.clip(1 - np.exp(-v / v1), 0, None))


def _hyperbolic(pts: np.ndarray, *, lambda_divisor: float = 3000.0, lambda_scale: float = 0.8,
                k_neighbors: int = 50, density_quantile: float = 0.99,
                floor_line: tuple[float, float] = (-1.0, 5.0), min_points: int = 20,
                start: tuple[float, float, float] = (8.0, 8.0, 0.25), maxiter: int = 1000) -> InSituProfile:
    inl = density_inliers(pts, k_neighbors=k_neighbors, density_quantile=density_quantile, floor_line=floor_line)
    p = pts[inl]
    if len(p) < min_points:
        raise ValueError(f"hyperbolic: {len(p)} inliers, at least {min_points} needed")
    lam = len(p) / lambda_divisor

    def above(curve_values: np.ndarray) -> int:
        return int(np.sum(p[:, 1] > curve_values))

    def f_linexp(x):
        a0, vm, v1 = x
        if not (a0 > 0 and vm > 0.5 and v1 > 1e-3):
            return _BAD
        val = lam * _linexp_area(a0, vm, v1) + above(linexp(p[:, 0], a0, vm, v1))
        return val if np.isfinite(val) else _BAD

    def f_hypexp(x):
        vm, a0, v0, v1 = x
        if not (vm > 0.5 and a0 > 0 and v0 > 0 and v1 > 1e-3):
            return _BAD
        area = quad(hypexp, 0, vm, args=(vm, a0, v0, v1), limit=50)[0]
        val = lam * lambda_scale * area + above(hypexp(p[:, 0], vm, a0, v0, v1))
        return val if np.isfinite(val) else _BAD

    lin = minimize(f_linexp, list(start), method="Nelder-Mead", options={"maxiter": maxiter}).x
    x0 = [lin[1] - 0.5, lin[0], lin[1], lin[2]]  # (vm, a0, v0, v1) started from LinExp
    vm, a0, v0, v1 = (float(x) for x in minimize(f_hypexp, x0, method="Nelder-Mead",
                                                  options={"maxiter": maxiter}).x)
    if not (vm > 0.5 and a0 > 0 and v0 > 0 and v1 > 0):
        raise ValueError("hyperbolic: the fit left the physiological domain")

    grid = np.linspace(0, vm, 400)
    peak = float(np.nanmax(hypexp(grid, vm, a0, v0, v1)))
    intercept = float(np.sqrt(abs((a0 / v0) * (2 * v0 + vm) * vm)))
    return InSituProfile(
        method="hyperbolic", a0=intercept, s0=vm, peak_acceleration=peak,
        params={"vm": vm, "a0": a0, "v0": v0, "v1": v1, "lambda": lam,
                "linexp_a0": float(lin[0]), "linexp_vm": float(lin[1]), "linexp_v1": float(lin[2])},
        curve=lambda x: hypexp(x, vm, a0, v0, v1),
        speed=p[:, 0], acceleration=p[:, 1],
    )


def _clean_for_bins(pts, min_speed, k_neighbors, density_quantile, floor_line, min_points=50):
    if len(pts) < min_points:
        raise ValueError(f"at least {min_points} points are needed (got {len(pts)})")
    p = pts[density_inliers(pts, k_neighbors=k_neighbors, density_quantile=density_quantile, floor_line=floor_line)]
    return p[p[:, 0] > min_speed]


def _tukey(pts, *, bin_width=0.1, min_speed=2.0, min_per_bin=5, k_neighbors=50,
           density_quantile=0.99, floor_line=(-1.0, 5.0)) -> InSituProfile:
    p = _clean_for_bins(pts, min_speed, k_neighbors, density_quantile, floor_line)
    edges = np.arange(np.floor(p[:, 0].min() / bin_width) * bin_width, p[:, 0].max() + bin_width, bin_width)
    vs, as_ = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        acc = p[(p[:, 0] >= lo) & (p[:, 0] < hi), 1]
        if acc.size < min_per_bin:
            continue
        q1, q3 = np.percentile(acc, [25, 75])
        acc = acc[(acc >= q1 - 1.5 * (q3 - q1)) & (acc <= q3 + 1.5 * (q3 - q1))]
        if acc.size:
            vs.append(lo + bin_width / 2)
            as_.append(acc.max())
    return _line_profile("tukey", np.array(vs), np.array(as_))


def _prediction_interval(pts, *, bin_width=0.1, min_speed=2.0, n_per_bin=2, t_value=2.0, max_iter=10,
                         k_neighbors=50, density_quantile=0.99, floor_line=(-1.0, 5.0)) -> InSituProfile:
    p = _clean_for_bins(pts, min_speed, k_neighbors, density_quantile, floor_line)
    edges = np.arange(np.floor(p[:, 0].min() / bin_width) * bin_width, p[:, 0].max() + bin_width, bin_width)
    sel = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = p[(p[:, 0] >= lo) & (p[:, 0] < hi)]
        if len(b) >= n_per_bin:
            sel.append(b[np.argsort(b[:, 1])[-n_per_bin:]])
    if not sel:
        raise ValueError("prediction_interval: no populated speed band")
    q = np.vstack(sel)
    for _ in range(max_iter):
        x, y = q[:, 0], q[:, 1]
        if len(x) <= 3:
            break
        slope, intercept = np.polyfit(x, y, 1)
        res = y - (slope * x + intercept)
        s = np.sqrt(np.sum(res**2) / (len(x) - 2))
        sxx = np.sum((x - x.mean()) ** 2)
        if sxx == 0:
            break
        keep = np.abs(res) <= t_value * s * np.sqrt(1 + 1 / len(x) + (x - x.mean()) ** 2 / sxx)
        if keep.all():
            break
        q = q[keep]
    return _line_profile("prediction_interval", q[:, 0], q[:, 1])


def fit_in_situ(
    velocity: ArrayLike,
    acceleration: ArrayLike,
    method: str = "linear",
    *,
    max_speed: float = 12.0,
    max_acceleration: float = 12.0,
    **options,
) -> InSituProfile:
    """Acceleration-speed profile from tracking data with the chosen method.

    Parameters
    ----------
    velocity, acceleration:
        Speed (m/s) and acceleration (m/s²) samples taken while accelerating,
        e.g. from :meth:`sprintfv.kalman.Kinematics.profile_points`.
    method:
        ``"linear"``, ``"hyperbolic"``, ``"tukey"`` or ``"prediction_interval"``.
    max_speed, max_acceleration:
        Samples above are dropped as implausible.
    **options:
        Method-specific settings, all with the thesis defaults:

        * linear: ``velocity_step=0.2``, ``n_per_bin=2``, ``velocity_threshold=None``
        * hyperbolic: ``lambda_divisor=3000`` (smaller = tighter envelope),
          ``lambda_scale=0.8``, ``k_neighbors=50``, ``density_quantile=0.99``,
          ``floor_line=(-1, 5)``, ``min_points=20``, ``start=(8, 8, 0.25)``
        * tukey: ``bin_width=0.1``, ``min_speed=2``, ``min_per_bin=5`` + density options
        * prediction_interval: ``bin_width=0.1``, ``min_speed=2``, ``n_per_bin=2``,
          ``t_value=2``, ``max_iter=10`` + density options
    """
    pts = _points(velocity, acceleration, max_speed, max_acceleration)
    funcs = {"linear": _linear, "hyperbolic": _hyperbolic, "tukey": _tukey,
             "prediction_interval": _prediction_interval}
    if method not in funcs:
        raise ValueError(f"unknown method {method!r}: choose among {METHODS}")
    if len(pts) < 3:
        raise ValueError("fewer than 3 valid points")
    return funcs[method](pts, **options)
