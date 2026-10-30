"""Force-velocity-power profile of sprint acceleration (Samozino et al. 2016).

From the kinematics of the mono-exponential model and the athlete's body
mass, the net horizontal force applied to the ground at each instant is

    F_H(t) = m · a(t) + F_air(t),      F_air = k · (v - v_wind)²

with the aerodynamic coefficient k from body mass and height (Arsac &
Locatelli 2002, as in Samozino et al. 2016). F_H decreases almost linearly
with velocity: its intercept is F0, its x-intercept V0, Pmax = F0·V0/4.
The ratio of forces RF = F_H / sqrt(F_H² + (m·g)²) tells how much of the total
force is oriented forward; its decrease with speed (DRF) describes technique.

Two methods:

``"samozino"`` (default)
    Linear regression of F_H on v, sampled every ``dt`` from 0 to the time
    the athlete reaches ``until`` × MSS. This is the original computation.
``"analytic"``
    F0 is the force at v = 0 and V0 the exact root of F_H(v) = 0. Identical
    to {shorts} ``create_FVP()``, used in the tests to check this package
    against the R implementation.
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
    only computed by the ``"samozino"`` method.
    """

    body_mass: float
    f0: float
    v0: float
    method: str
    rf_max: float | None = None
    drf: float | None = None
    k_air: float = 0.0
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
    slope, f0 = np.polyfit(v, f_h, 1)
    v0 = -f0 / slope

    rf = 100.0 * f_h / np.sqrt(f_h**2 + (body_mass * G) ** 2)
    m = t >= rf_from
    drf = np.polyfit(v[m], rf[m], 1)[0] if m.sum() > 2 else float("nan")
    return ForceVelocityProfile(
        body_mass=body_mass, f0=float(f0), v0=float(v0), method=method,
        rf_max=float(rf[m].max()) if m.any() else None, drf=float(drf), k_air=k,
        table={"time": t, "velocity": v, "acceleration": a, "force": f_h, "power": f_h * v, "rf": rf},
    )


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
