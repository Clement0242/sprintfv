"""Mono-exponential model of a maximal sprint from a standing start.

The whole package rests on one equation (Furusawa et al. 1927; Clark et al. 2017):

    v(t) = MSS * (1 - exp(-t / tau)),      tau = MSS / MAC

MSS is the maximal sprinting speed (m/s), MAC the maximal acceleration (m/s²),
i.e. the slope of the velocity curve at t = 0. Everything else (distance,
acceleration, time to cover a distance) follows in closed form, so a profile
can be read at any instant or any distance without numerical integration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import lambertw


def _tau(mss: float, mac: float) -> float:
    if mss <= 0 or mac <= 0:
        raise ValueError(f"MSS and MAC must be positive (got MSS={mss}, MAC={mac})")
    return mss / mac


def velocity_at_time(t: ArrayLike, mss: float, mac: float) -> np.ndarray:
    """Velocity (m/s) at time ``t`` (s)."""
    tau = _tau(mss, mac)
    return mss * (1.0 - np.exp(-np.asarray(t, dtype=float) / tau))


def acceleration_at_time(t: ArrayLike, mss: float, mac: float) -> np.ndarray:
    """Acceleration (m/s²) at time ``t`` (s): MAC at the start, decaying to 0."""
    tau = _tau(mss, mac)
    return mac * np.exp(-np.asarray(t, dtype=float) / tau)


def distance_at_time(t: ArrayLike, mss: float, mac: float) -> np.ndarray:
    """Distance covered (m) at time ``t`` (s): the integral of v(t)."""
    tau = _tau(mss, mac)
    t = np.asarray(t, dtype=float)
    return mss * (t + tau * np.exp(-t / tau)) - mss * tau


def time_at_distance(d: ArrayLike, mss: float, mac: float) -> np.ndarray:
    """Time (s) needed to cover distance ``d`` (m).

    Inverting x(t) has no elementary solution, but it has an exact one with the
    principal branch of the Lambert W function. The argument lies in [-1/e, 0),
    where W0 is real, so the result is exact rather than iterated.
    """
    tau = _tau(mss, mac)
    d = np.asarray(d, dtype=float)
    # At d = 0 the argument is the branch point -1/e itself, where scipy's W0
    # returns NaN in floating point: the answer there is simply t = 0.
    arg = np.maximum(-np.exp(-d / (mss * tau) - 1.0), -np.nextafter(np.exp(-1.0), 0))
    t = tau * np.real(lambertw(arg, k=0)) + d / mss + tau
    return np.where(d <= 0, 0.0, t)


def velocity_at_distance(d: ArrayLike, mss: float, mac: float) -> np.ndarray:
    """Velocity (m/s) when the athlete reaches distance ``d`` (m)."""
    return velocity_at_time(time_at_distance(d, mss, mac), mss, mac)


def acceleration_at_distance(d: ArrayLike, mss: float, mac: float) -> np.ndarray:
    """Acceleration (m/s²) when the athlete reaches distance ``d`` (m)."""
    return acceleration_at_time(time_at_distance(d, mss, mac), mss, mac)


def time_at_velocity(v: ArrayLike, mss: float, mac: float) -> np.ndarray:
    """Time (s) at which velocity ``v`` is reached (``v`` must be below MSS)."""
    tau = _tau(mss, mac)
    v = np.asarray(v, dtype=float)
    if np.any(v >= mss):
        raise ValueError("velocity must be strictly below MSS: the model only reaches MSS asymptotically")
    return -tau * np.log(1.0 - v / mss)
