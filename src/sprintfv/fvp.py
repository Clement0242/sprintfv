"""Force-velocity-power profile of sprint acceleration (Samozino et al. 2016).

From the kinematics of the mono-exponential model and the athlete's body
mass, the net horizontal force applied to the ground at each instant is

    F_H(t) = m · a(t) + F_air(t),      F_air = k · (v - v_wind)²

with the aerodynamic coefficient k from body mass and height (Arsac &
Locatelli 2002, as in Samozino et al. 2016). F_H decreases almost linearly
with velocity: its intercept is F0, its x-intercept V0, Pmax = F0·V0/4.
The ratio of forces RF = F_H / sqrt(F_H² + (m·g)²) tells how much of the total
force is oriented forward; its decrease with speed (DRF) describes technique.

Two routes to the same profile:

* **Model-based** (:func:`force_velocity_profile`), the "simple method" of
  Samozino et al. (ISB 2013; Scand J Med Sci Sports 2016): fit the
  mono-exponential model to split times or a radar trace, derive F_H(t) from
  the model, then the linear F-v regression. Works with as few as 5 splits.

  ``"samozino"`` (default)
      Linear regression of F_H on v, sampled every ``dt`` from 0 to the time
      the athlete reaches ``until`` × MSS. This is the original computation.
  ``"analytic"``
      F0 is the force at v = 0 and V0 the exact root of F_H(v) = 0. Identical
      to {shorts} ``create_FVP()``, used in the tests to check this package
      against the R implementation.

* **Direct** (:func:`force_velocity_from_trace`): when a sensor gives the
  speed itself (radar, laser, LPS/GPS), F_H is computed from the *measured*
  speed and its derivative, and the linear F-v regression is fitted to those
  points. No exponential shape is imposed, and the R² of the regression tests
  how linear the athlete's profile really is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from . import model

G = 9.81


def air_resistance_coefficient(
    body_mass: float,
    body_height: float = 1.75,
    air_temperature: float = 25.0,
    barometric_pressure: float = 760.0,
) -> float:
    """k (kg/m) so that F_air = k · v². Temperature in °C, pressure in mmHg."""
    rho = 1.293 * (barometric_pressure / 760.0) * (273.0 / (273.0 + air_temperature))
    frontal_area = 0.2025 * body_height**0.725 * body_mass**0.425 * 0.266
    return 0.5 * rho * frontal_area * 0.9


@dataclass
class ForceVelocityProfile:
    """Force-velocity-power profile.

    Forces in N, velocities in m/s, powers in W; ``*_rel`` per kg of body mass.
    ``sfv_rel`` is the slope of the relative profile (N/kg per m/s, negative).
    ``rf_max`` (%) and ``drf`` (% per m/s) describe force orientation; they are
    not computed by the ``"analytic"`` method. ``r2`` is the R² of the linear
    F-v regression: close to 1 by construction for the model-based route,
    an actual test of linearity for the direct route.
    """

    body_mass: float
    f0: float
    v0: float
    method: str
    rf_max: float | None = None
    drf: float | None = None
    k_air: float = 0.0
    r2: float | None = None
    table: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def f0_rel(self) -> float:
        return self.f0 / self.body_mass

    @property
    def pmax(self) -> float:
        return self.f0 * self.v0 / 4.0

    @property
    def pmax_rel(self) -> float:
        return self.pmax / self.body_mass

    @property
    def sfv_rel(self) -> float:
        return -self.f0_rel / self.v0

    def summary(self) -> str:
        lines = [
            f"Force-velocity profile ({self.method}, {self.body_mass:g} kg)",
            f"  F0    {self.f0:8.1f} N    ({self.f0_rel:.2f} N/kg)",
            f"  V0    {self.v0:8.2f} m/s",
            f"  Pmax  {self.pmax:8.0f} W    ({self.pmax_rel:.2f} W/kg)",
            f"  SFV   {self.sfv_rel:8.3f} N/kg per m/s",
        ]
        if self.rf_max is not None:
            lines += [f"  RFmax {self.rf_max:8.1f} %", f"  DRF   {self.drf:8.2f} % per m/s"]
        if self.r2 is not None:
            lines.append(f"  R²    {self.r2:8.3f}   (linearity of the F-v regression)")
        return "\n".join(lines)


def force_velocity_profile(
    mss: float,
    mac: float,
    body_mass: float,
    *,
    body_height: float = 1.75,
    air_temperature: float = 25.0,
    barometric_pressure: float = 760.0,
    wind_velocity: float = 0.0,
    method: str = "samozino",
    dt: float = 0.01,
    until: float = 0.99,
    rf_from: float = 0.3,
) -> ForceVelocityProfile:
    """Compute the F-V profile from MSS and MAC (e.g. ``fit.mss, fit.mac``).

    Parameters
    ----------
    until:
        ``"samozino"`` only: the regression uses the sprint until velocity
        reaches this fraction of MSS (the asymptote itself is never reached).
    rf_from:
        RF is computed from this time (s): before it, the athlete is still in
        the starting posture and RF is not meaningful (Samozino et al. 2016).
    """
    if body_mass <= 0:
        raise ValueError("body_mass must be positive")
    k = air_resistance_coefficient(body_mass, body_height, air_temperature, barometric_pressure)

    if method == "analytic":
        f0 = body_mass * mac + k * np.sign(-wind_velocity) * wind_velocity**2
        b = body_mass * mac + 2 * k * mss * wind_velocity
        disc = b**2 - 4 * k * mss**2 * (body_mass * mac + k * wind_velocity**2)
        v0 = (b - np.sqrt(disc)) / (2 * k * mss)
        return ForceVelocityProfile(body_mass=body_mass, f0=float(f0), v0=float(v0), method=method, k_air=k)

    if method != "samozino":
        raise ValueError(f"unknown method {method!r}: use 'samozino' or 'analytic'")
    if not 0 < until < 1:
        raise ValueError("until must be in (0, 1)")

    t_end = float(model.time_at_velocity(until * mss, mss, mac))
    t = np.arange(0.0, t_end + dt / 2, dt)
    v = model.velocity_at_time(t, mss, mac)
    a = model.acceleration_at_time(t, mss, mac)
    rel_v = v - wind_velocity
    f_h = body_mass * a + k * np.sign(rel_v) * rel_v**2
    return _linear_profile(t, v, a, f_h, body_mass, k, method, rf_from)


def _linear_profile(t, v, a, f_h, body_mass, k, method, rf_from) -> ForceVelocityProfile:
    """Linear F-v regression, ratio of forces, and the profile object."""
    slope, f0 = np.polyfit(v, f_h, 1)
    if slope >= 0:
        raise ValueError("force does not decrease with velocity: is this an acceleration phase?")
    v0 = -f0 / slope
    pred = f0 + slope * v
    r2 = 1 - np.sum((f_h - pred) ** 2) / np.sum((f_h - f_h.mean()) ** 2)

    rf = 100.0 * f_h / np.sqrt(f_h**2 + (body_mass * G) ** 2)
    m = (t - t[0]) >= rf_from
    drf = np.polyfit(v[m], rf[m], 1)[0] if m.sum() > 2 else float("nan")
    return ForceVelocityProfile(
        body_mass=body_mass, f0=float(f0), v0=float(v0), method=method,
        rf_max=float(rf[m].max()) if m.any() else None, drf=float(drf), k_air=k, r2=float(r2),
        table={"time": t, "velocity": v, "acceleration": a, "force": f_h, "power": f_h * v, "rf": rf},
    )


def force_velocity_from_trace(
    time: np.ndarray,
    velocity: np.ndarray,
    body_mass: float,
    *,
    acceleration: np.ndarray | None = None,
    body_height: float = 1.75,
    air_temperature: float = 25.0,
    barometric_pressure: float = 760.0,
    wind_velocity: float = 0.0,
    smoothing_s: float = 0.5,
    v_start: float = 0.5,
    until: float = 0.95,
    end_time: float | None = None,
    rf_from: float = 0.3,
) -> ForceVelocityProfile:
    """Force-velocity profile straight from a measured speed trace (direct route).

    No mono-exponential model: at every sample of the acceleration phase,
    F_H = m · a + F_air with the *measured* speed, then the linear F-v
    regression. Use it when a sensor gives the speed itself (radar, laser,
    LPS/GPS through :func:`sprintfv.positions_to_kinematics`).

    Parameters
    ----------
    time, velocity:
        One sprint from a standing start (s, m/s), cleaned beforehand
        (:func:`sprintfv.clean_radar` for a radar).
    acceleration:
        Acceleration from the sensor or a Kalman filter (m/s²). When omitted,
        it is the derivative of a Savitzky-Golay fit over ``smoothing_s``.
    v_start:
        Samples below this speed (start posture, first push) are left out.
    until:
        Samples above this fraction of the peak speed are left out: near the
        plateau, acceleration is tiny and dominated by noise.
    end_time:
        Time at which the acceleration phase ends; found with
        :func:`sprintfv.acceleration_end` when omitted.
    """
    from scipy.signal import savgol_filter

    from .signal import acceleration_end

    if body_mass <= 0:
        raise ValueError("body_mass must be positive")
    t = np.asarray(time, dtype=float).ravel()
    v = np.asarray(velocity, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(v)
    order = np.argsort(t[ok])
    t, v = t[ok][order], v[ok][order]
    if acceleration is None:
        dt = float(np.median(np.diff(t)))
        win = max(5, int(round(smoothing_s / dt)) | 1)
        if win >= t.size:
            raise ValueError("trace too short for the smoothing window")
        a = savgol_filter(v, win, 2, deriv=1, delta=dt)
    else:
        a = np.asarray(acceleration, dtype=float).ravel()[ok][order]

    t_end = end_time if end_time is not None else t[acceleration_end(t, v)]
    vmax = np.nanmax(v[t <= t_end]) if np.any(t <= t_end) else np.nanmax(v)
    m = (t <= t_end) & (v >= v_start) & (v <= until * vmax) & np.isfinite(a)
    if m.sum() < 10:
        raise ValueError(f"only {int(m.sum())} samples in the acceleration phase: at least 10 are needed")
    t, v, a = t[m], v[m], a[m]

    k = air_resistance_coefficient(body_mass, body_height, air_temperature, barometric_pressure)
    rel_v = v - wind_velocity
    f_h = body_mass * a + k * np.sign(rel_v) * rel_v**2
    return _linear_profile(t, v, a, f_h, body_mass, k, "direct", rf_from)


# ------------------------------------------------------------ optimal profile

def _time_to_distance(f0_rel: float, v0: float, distance: float, k_rel: float) -> float:
    """Time to cover ``distance`` for a linear F-V profile, by integrating
    dv/dt = F0_rel · (1 - v/V0) - k_rel · v²."""
    def rhs(_, y):
        return [y[1], f0_rel * (1 - y[1] / v0) - k_rel * y[1] ** 2]

    def reached(_, y):
        return y[0] - distance
    reached.terminal, reached.direction = True, 1

    sol = solve_ivp(rhs, (0, 60), [0.0, 0.0], events=reached, rtol=1e-8, atol=1e-10, max_step=0.05)
    return float(sol.t_events[0][0]) if sol.t_events[0].size else np.inf


@dataclass
class OptimalProfile:
    """Optimal F-V profile for a distance at the athlete's Pmax (Samozino et al. 2022).

    ``fv_imbalance`` = 100 · (SFV / SFV_opt - 1): positive = force-oriented
    profile (train velocity), negative = velocity-oriented (train force),
    close to 0 = balanced. ``time_gain`` (s) is what the optimal profile would
    save over the distance at the same Pmax.
    """

    distance: float
    sfv_rel: float
    sfv_opt: float
    f0_opt_rel: float
    v0_opt: float
    time_actual: float
    time_optimal: float

    @property
    def fv_imbalance(self) -> float:
        return 100.0 * (self.sfv_rel / self.sfv_opt - 1.0)

    @property
    def time_gain(self) -> float:
        return self.time_actual - self.time_optimal


def optimal_profile(
    fvp: ForceVelocityProfile,
    distance: float,
    *,
    air_resistance: bool = True,
    sfv_bounds: tuple[float, float] = (-3.0, -0.2),
) -> OptimalProfile:
    """Slope of the F-V profile that minimises the time over ``distance``, at
    the athlete's own Pmax. The sprint is simulated numerically, so air
    resistance can be kept (``air_resistance=True``) or ignored."""
    p = fvp.pmax_rel
    k_rel = fvp.k_air / fvp.body_mass if air_resistance else 0.0

    def time_for(sfv: float) -> float:
        f0 = np.sqrt(-4 * p * sfv)
        return _time_to_distance(f0, f0 / -sfv, distance, k_rel)

    res = minimize_scalar(time_for, bounds=sfv_bounds, method="bounded", options={"xatol": 1e-6})
    sfv_opt = float(res.x)
    f0_opt = float(np.sqrt(-4 * p * sfv_opt))
    return OptimalProfile(
        distance=distance, sfv_rel=fvp.sfv_rel, sfv_opt=sfv_opt,
        f0_opt_rel=f0_opt, v0_opt=f0_opt / -sfv_opt,
        time_actual=_time_to_distance(fvp.f0_rel, fvp.v0, distance, k_rel),
        time_optimal=float(res.fun),
    )


__all__ = [
    "ForceVelocityProfile", "OptimalProfile", "air_resistance_coefficient",
    "force_velocity_profile", "optimal_profile",
]
