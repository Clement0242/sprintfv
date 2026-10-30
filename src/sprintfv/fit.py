"""Estimating MSS and MAC from the three usual sources of sprint data.

* Radar or laser gun: velocity sampled over time      -> :func:`fit_radar`
* Timing gates: times at a few distances               -> :func:`fit_splits`
* GPS / LPS in training or matches (no sprint test)    -> :mod:`sprintfv.insitu`

Every fit returns a :class:`SprintFit`, which carries the parameters, their
standard errors, the corrections that were estimated, and goodness-of-fit
numbers. The non-linear fits mirror the models of the R package {shorts}
(Jovanović, MIT licence), so the two can be compared number for number; the
test suite reproduces the values published in its README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from . import model


@dataclass(frozen=True)
class SprintFit:
    """Result of a sprint fit.

    Attributes
    ----------
    mss, mac:
        Maximal sprinting speed (m/s) and maximal acceleration (m/s²).
    tau:
        MSS / MAC (s), the time constant of the exponential.
    pmax_rel:
        MSS * MAC / 4 (W/kg), maximal relative power *without* air resistance.
        :func:`sprintfv.force_velocity_profile` gives the proper value.
    corrections:
        Estimated corrections, e.g. ``{"time_correction": -0.01}``.
    std_errors:
        Standard error of each estimated parameter (from the Jacobian).
    method:
        ``"radar"`` or ``"splits"``.
    predictor, observed, predicted:
        The data the model was fitted on, and its prediction.
    """

    mss: float
    mac: float
    method: str
    corrections: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    predictor: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    observed: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    predicted: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def tau(self) -> float:
        return self.mss / self.mac

    @property
    def pmax_rel(self) -> float:
        return self.mss * self.mac / 4.0

    @property
    def residuals(self) -> np.ndarray:
        return self.observed - self.predicted

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))

    @property
    def r2(self) -> float:
        ss_res = np.sum(self.residuals**2)
        ss_tot = np.sum((self.observed - self.observed.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot)

    def velocity_at_time(self, t: ArrayLike) -> np.ndarray:
        return model.velocity_at_time(t, self.mss, self.mac)

    def time_at_distance(self, d: ArrayLike) -> np.ndarray:
        return model.time_at_distance(d, self.mss, self.mac)

    def splits(self, distances: ArrayLike = (5, 10, 20, 30, 40)) -> dict[float, float]:
        """Predicted split times at the given distances."""
        d = np.asarray(distances, dtype=float)
        return dict(zip(d.tolist(), self.time_at_distance(d).tolist()))

    def summary(self) -> str:
        lines = [
            f"Sprint profile ({self.method})",
            f"  MSS   {self.mss:8.3f} m/s",
            f"  MAC   {self.mac:8.3f} m/s²",
            f"  TAU   {self.tau:8.3f} s",
            f"  PMAX  {self.pmax_rel:8.3f} W/kg (no air resistance)",
        ]
        for k, v in self.corrections.items():
            lines.append(f"  {k} {v:+.4f}")
        if self.observed.size:
            lines.append(f"  R²    {self.r2:8.5f}   RMSE {self.rmse:.4f}")
        return "\n".join(lines)


def _clean(x: ArrayLike, y: ArrayLike, names: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"{names[0]} and {names[1]} must have the same length ({x.size} vs {y.size})")
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        raise ValueError("at least 3 valid observations are needed")
    return x[keep], y[keep]


def _std_errors(names: list[str], pcov: np.ndarray) -> dict[str, float]:
    se = np.sqrt(np.clip(np.diag(pcov), 0, None))
    return dict(zip(names, se.tolist()))


def fit_radar(
    time: ArrayLike,
    velocity: ArrayLike,
    *,
    time_correction: bool = True,
    use_observed_mss: bool = False,
) -> SprintFit:
    """Fit the mono-exponential model to radar or laser velocity-time data.

    Parameters
    ----------
    time, velocity:
        Samples of one sprint from a standing start, in s and m/s. Trim the
        deceleration after the finish first (see :mod:`sprintfv.signal`).
    time_correction:
        Also estimate a time shift ``TC`` so that v(t) = model(t - TC). The
        moment the radar starts is rarely the moment the athlete starts.
    use_observed_mss:
        Fix MSS to the highest observed velocity instead of estimating it, as
        in the original method of Samozino et al. (2016).
    """
    t, v = _clean(time, velocity, ("time", "velocity"))
    v_peak = float(v.max())

    if use_observed_mss:
        def f(tt, mac, tc=0.0):
            return model.velocity_at_time(tt - tc, v_peak, mac)
        p0, names = ([7.0, 0.0], ["mac", "time_correction"]) if time_correction else ([7.0], ["mac"])
        lower = [1e-6] + ([-np.inf] if time_correction else [])
    else:
        def f(tt, mss, mac, tc=0.0):
            return model.velocity_at_time(tt - tc, mss, mac)
        p0 = [max(v_peak, 1.0), 7.0] + ([0.0] if time_correction else [])
        names = ["mss", "mac"] + (["time_correction"] if time_correction else [])
        lower = [1e-6, 1e-6] + ([-np.inf] if time_correction else [])

    popt, pcov = curve_fit(f, t, v, p0=p0, bounds=(lower, np.inf), maxfev=20000)
    est = dict(zip(names, popt.tolist()))
    mss = v_peak if use_observed_mss else est["mss"]
    corrections = {"time_correction": est["time_correction"]} if time_correction else {}
    return SprintFit(
        mss=mss, mac=est["mac"], method="radar", corrections=corrections,
        std_errors=_std_errors(names, pcov),
        predictor=t, observed=v, predicted=f(t, *popt),
    )


def fit_splits(
    distance: ArrayLike,
    time: ArrayLike,
    *,
    correction: str | None = None,
) -> SprintFit:
    """Fit the model to timing-gate split times.

    Parameters
    ----------
    distance, time:
        Gate distances (m) and the times (s) at which they were crossed.
    correction:
        * ``None``: the clock starts exactly when the athlete starts.
        * ``"time"``: estimate a constant time offset ``TC`` added to every
          split (reaction to the start signal, trigger by the back foot...).
        * ``"flying_distance"``: estimate a flying-start distance ``FD`` the
          athlete covers before the first gate starts the clock.

    Split times are sensitive to how the clock starts: with few gates, an
    uncorrected fit inflates MAC. Comparing the corrections is the honest way
    to see how much the result depends on that assumption.
    """
    d, t = _clean(distance, time, ("distance", "time"))

    if correction is None:
        def f(dd, mss, mac):
            return model.time_at_distance(dd, mss, mac)
        p0, names, lower = [7.0, 7.0], ["mss", "mac"], [1e-6, 1e-6]
    elif correction == "time":
        def f(dd, mss, mac, tc):
            return model.time_at_distance(dd, mss, mac) + tc
        p0, names, lower = [7.0, 7.0, 0.0], ["mss", "mac", "time_correction"], [1e-6, 1e-6, -np.inf]
    elif correction == "flying_distance":
        def f(dd, mss, mac, fd):
            return model.time_at_distance(dd + fd, mss, mac) - model.time_at_distance(fd, mss, mac)
        p0, names, lower = [7.0, 7.0, 0.1], ["mss", "mac", "flying_distance"], [1e-6, 1e-6, 0.0]
    else:
        raise ValueError(f"unknown correction {correction!r}: use None, 'time' or 'flying_distance'")

    if d.size < len(p0):
        raise ValueError(f"{len(p0)} parameters need at least {len(p0)} splits (got {d.size})")

    popt, pcov = curve_fit(f, d, t, p0=p0, bounds=(lower, np.inf), maxfev=20000)
    est = dict(zip(names, popt.tolist()))
    corrections = {k: v for k, v in est.items() if k not in ("mss", "mac")}
    return SprintFit(
        mss=est["mss"], mac=est["mac"], method="splits", corrections=corrections,
        std_errors=_std_errors(names, pcov),
        predictor=d, observed=t, predicted=f(d, *popt),
    )

