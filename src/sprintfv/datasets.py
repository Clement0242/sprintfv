"""Sample datasets from the R package {shorts} (Jovanović, MIT licence).

* ``jb_morin``: one radar sprint shared by Jean-Benoît Morin with his
  profiling spreadsheet (75 kg, 1.72 m, 25 °C, 760 mmHg, no wind).
* ``radar_gun``: simulated radar sprints of 5 athletes (100 Hz).
* ``split_times``: timing-gate splits of the same 5 athletes.
* ``lps_session``: x/y positions of one player over a ~95 min basketball
  practice, from a local positioning system at 20 Hz.

Returned as dictionaries of NumPy arrays, so pandas is not required.
"""

from __future__ import annotations

import csv
import gzip
import io
from importlib.resources import files

import numpy as np


def _read(name: str) -> dict[str, np.ndarray]:
    raw = files("sprintfv.data").joinpath(name).read_bytes()
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    out = {}
    for key in rows[0]:
        col = [r[key] for r in rows]
        try:
            out[key] = np.array(col, dtype=float)
        except ValueError:
            out[key] = np.array(col)
    return out


def _athlete(data: dict[str, np.ndarray], athlete: str | None) -> dict[str, np.ndarray]:
    if athlete is None:
        return data
    m = data["athlete"] == athlete
    if not m.any():
        raise KeyError(f"unknown athlete {athlete!r}: {sorted(set(data['athlete']))}")
    return {k: v[m] for k, v in data.items()}


def load_jb_morin() -> dict[str, np.ndarray]:
    return _read("jb_morin.csv")


def load_radar_gun(athlete: str | None = None) -> dict[str, np.ndarray]:
    return _athlete(_read("radar_gun_data.csv"), athlete)


def load_split_times(athlete: str | None = None) -> dict[str, np.ndarray]:
    return _athlete(_read("split_times.csv"), athlete)


def load_lps_session() -> dict[str, np.ndarray]:
    return _read("lps_session.csv.gz")
