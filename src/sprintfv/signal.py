"""Cleaning a raw radar (or laser) velocity trace before fitting.

Radar guns drop samples (zeros), produce spikes when the beam leaves the
athlete's back, and keep recording after the finish. The chain below is the
one used in production at a national federation, every step configurable:

1. sort by time and start the clock at 0;
2. convert km/h to m/s when the peak is implausible in m/s (> 15);
3. replace *isolated* zeros by the mean of their neighbours (real stops kept);
4. remove outliers beyond ``threshold`` rolling standard deviations (as NaN);
5. zero-phase Butterworth low-pass, NaNs bridged only for filtering and put back.

Then :func:`acceleration_end` finds where the acceleration phase ends, so that
the plateau and the deceleration do not bias the fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal as sps


@dataclass
class CleanTrace:
    time: np.ndarray
    raw: np.ndarray
    velocity: np.ndarray
    unit: str
    zeros_replaced: int
    outliers_removed: int
    sampling_rate: float


def _rolling(x: np.ndarray, window: int, func) -> np.ndarray:
    half = window // 2
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        w = x[max(0, i - half): i + half + 1]
        w = w[np.isfinite(w)]
        if w.size >= 2:
            out[i] = func(w)
    return out


def clean_radar(
    time: ArrayLike,
    velocity: ArrayLike,
    *,
    kmh_threshold: float = 15.0,
    zero_window: int = 5,
    outlier_window: int = 15,
    outlier_threshold: float = 2.0,
    cutoff_hz: float | None = 0.9,
    filter_order: int = 4,
    sampling_rate: float | None = None,
) -> CleanTrace:
    """Apply the cleaning chain and return a :class:`CleanTrace`.

    Parameters
    ----------
    kmh_threshold:
        Peak above which the trace is assumed to be in km/h.
    zero_window:
        Neighbours on each side inspected around a zero. A zero is replaced
        only if at least 3 neighbours are valid, their mean exceeds 0.5 m/s
        and fewer than half of them are zeros.
    outlier_window, outlier_threshold:
        Rolling window (samples) and number of standard deviations.
    cutoff_hz, filter_order:
        Butterworth low-pass (``cutoff_hz=None`` disables it).
    sampling_rate:
        Hz. Estimated from the median time step when omitted.
    """
    t = np.asarray(time, dtype=float).ravel()
    v = np.asarray(velocity, dtype=float).ravel()
    order = np.argsort(t, kind="stable")
    t, v = t[order] - t[order][0], v[order]
    raw = v.copy()

    unit = "ms"
    if np.nanmax(v) > kmh_threshold:
        v, unit = v / 3.6, "kmh"

    s = v.copy()
    replaced = 0
    for i in np.flatnonzero(v == 0):
        nb = np.concatenate((v[max(0, i - zero_window): i], v[i + 1: i + 1 + zero_window]))
        nb = nb[np.isfinite(nb)]
        if nb.size >= 3 and nb.mean() > 0.5 and np.mean(nb == 0) < 0.5:
            s[i] = nb.mean()
            replaced += 1

    mean = _rolling(s, outlier_window, np.mean)
    std = _rolling(s, outlier_window, np.std)
    with np.errstate(invalid="ignore"):
        out = np.isfinite(std) & (std > 0) & (np.abs(s - mean) > outlier_threshold * std)
    s[out] = np.nan

    fs = sampling_rate or 1.0 / np.median(np.diff(t))
    if cutoff_hz is not None:
        nan = np.isnan(s)
        filled = np.interp(np.arange(s.size), np.flatnonzero(~nan), s[~nan]) if nan.any() else s
        sos = sps.butter(filter_order, cutoff_hz, "lp", fs=fs, output="sos")
        s = sps.sosfiltfilt(sos, filled)
        s[nan] = np.nan

    return CleanTrace(time=t, raw=raw, velocity=s, unit=unit, zeros_replaced=replaced,
                      outliers_removed=int(out.sum()), sampling_rate=float(fs))


def acceleration_end(
    time: ArrayLike,
    velocity: ArrayLike,
    *,
    window_s: float = 0.25,
    a_threshold: float = 0.20,
    v_fraction: float = 0.98,
) -> int:
    """Index where the acceleration phase ends.

    The first sample after which, for ``window_s`` seconds, the smoothed
    acceleration (Savitzky-Golay derivative) stays below ``a_threshold`` while
    velocity stays above ``v_fraction`` of its peak. Falls back to the index of
    the peak velocity.
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(velocity, dtype=float)
    ok = np.isfinite(v)
    v = np.interp(np.arange(v.size), np.flatnonzero(ok), v[ok])
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(window_s / dt)))
    win = min(max(5, w | 1), v.size - (1 - v.size % 2))
    acc = sps.savgol_filter(v, win, 3, deriv=1, delta=dt) if win > 3 else np.gradient(v, t)
    i_peak = int(np.argmax(v))
    vmax = v[i_peak]
    # Searched from the start: on a noisy plateau the peak itself can come late,
    # long after the acceleration has ended.
    for i in range(0, v.size - w + 1):
        seg = slice(i, i + w)
        if np.all(acc[seg] < a_threshold) and np.all(v[seg] > v_fraction * vmax):
            return i
    return i_peak
