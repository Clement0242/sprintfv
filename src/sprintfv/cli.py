"""Command line: ``sprintfv radar|splits|insitu FILE.csv [options]``.

Reads a CSV with a header row and prints the profile. Examples::

    sprintfv radar sprint.csv --mass 75 --height 1.80
    sprintfv splits gates.csv --mass 60 --correction time
    sprintfv insitu session.csv --method hyperbolic --process-var 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from . import fit, fvp, insitu, kalman, signal


def _columns(path: str, *names: str) -> list[np.ndarray]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: empty file")
    missing = [n for n in names if n not in rows[0]]
    if missing:
        sys.exit(f"{path}: missing column(s) {missing}; available: {list(rows[0])}")
    return [np.array([float(r[n]) if r[n] not in ("", "NA") else np.nan for r in rows]) for n in names]


def _profile(args, f: fit.SprintFit, trace=None) -> None:
    print(f.summary())
    if args.mass:
        air = dict(body_height=args.height, air_temperature=args.temperature, barometric_pressure=args.pressure)
        if args.fv_method == "direct":
            if trace is None:
                sys.exit("--fv-method direct needs a speed trace (radar), not split times")
            p = fvp.force_velocity_from_trace(*trace, args.mass, **air)
        else:
            p = fvp.force_velocity_profile(f.mss, f.mac, args.mass, method=args.fv_method, **air)
        print(p.summary())
        o = None
        if args.distance:
            o = fvp.optimal_profile(p, args.distance)
            print(f"Optimal profile over {args.distance:g} m: SFV_opt {o.sfv_opt:.3f}, "
                  f"FV imbalance {o.fv_imbalance:+.0f} %, potential gain {o.time_gain:.3f} s")
        if args.report:
            from .report import sprint_report
            path = sprint_report(f, p, args.report, athlete=args.athlete, optimal=o, trace=trace,
                                 body_height=args.height, lang=args.lang, organisation=args.organisation,
                                 logo=args.logo, test=Path(args.file).name)
            print(f"Report written: {path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sprintfv", description="Sprint and force-velocity profiling.")
    sub = ap.add_subparsers(dest="command", required=True)

    def athlete(p):
        p.add_argument("--mass", type=float, help="body mass (kg): adds the force-velocity profile")
        p.add_argument("--height", type=float, default=1.75, help="body height (m)")
        p.add_argument("--temperature", type=float, default=25.0, help="air temperature (°C)")
        p.add_argument("--pressure", type=float, default=760.0, help="barometric pressure (mmHg)")
        p.add_argument("--fv-method", choices=["samozino", "analytic", "direct"], default="samozino",
                       help="direct: force from the measured speed itself, no exponential model (radar only)")
        p.add_argument("--distance", type=float, help="distance (m) for the optimal profile")
        p.add_argument("--report", metavar="PDF", help="also write a designed PDF report (needs sprintfv[report])")
        p.add_argument("--athlete", help="athlete name printed on the report")
        p.add_argument("--lang", choices=["fr", "en"], default="fr", help="report language")
        p.add_argument("--organisation", help="your organisation, printed on the report")
        p.add_argument("--logo", help="your logo (PNG/SVG), printed on the report")

    r = sub.add_parser("radar", help="radar or laser gun: time, velocity")
    r.add_argument("file")
    r.add_argument("--time-col", default="time")
    r.add_argument("--velocity-col", default="velocity")
    r.add_argument("--clean", action="store_true", help="clean the raw trace and cut it at the end of acceleration")
    r.add_argument("--observed-mss", action="store_true", help="fix MSS to the peak velocity")
    athlete(r)

    s = sub.add_parser("splits", help="timing gates: distance, time")
    s.add_argument("file")
    s.add_argument("--distance-col", default="distance")
    s.add_argument("--time-col", default="time")
    s.add_argument("--correction", choices=["time", "flying_distance"])
    athlete(s)

    i = sub.add_parser("insitu", help="GPS/LPS positions (time, x, y) or speed/acceleration")
    i.add_argument("file")
    i.add_argument("--method", choices=insitu.METHODS, default="linear")
    i.add_argument("--time-col", default="time")
    i.add_argument("--x-col", default="x")
    i.add_argument("--y-col", default="y")
    i.add_argument("--speed-col", help="use speed/acceleration columns instead of positions")
    i.add_argument("--acceleration-col")
    i.add_argument("--dt", type=float, default=0.05, help="resampling step (s)")
    i.add_argument("--order", type=int, default=4, help="Kalman states (3 or 4)")
    i.add_argument("--process-var", type=float, default=5.0, help="Kalman process noise variance")
    i.add_argument("--measurement-std", type=float, default=0.05, help="position noise (m)")
    i.add_argument("--prefilter-hz", type=float, default=1.0, help="Butterworth cut-off, 0 to disable")

    args = ap.parse_args(argv)
    if args.command == "radar":
        t, v = _columns(args.file, args.time_col, args.velocity_col)
        if args.clean:
            c = signal.clean_radar(t, v)
            end = signal.acceleration_end(c.time, c.velocity)
            ok = np.isfinite(c.velocity[:end + 1])
            t, v = c.time[:end + 1][ok], c.velocity[:end + 1][ok]
        _profile(args, fit.fit_radar(t, v, use_observed_mss=args.observed_mss), trace=(t, v))
    elif args.command == "splits":
        d, t = _columns(args.file, args.distance_col, args.time_col)
        _profile(args, fit.fit_splits(d, t, correction=args.correction))
    else:
        if args.speed_col:
            v, a = _columns(args.file, args.speed_col, args.acceleration_col or "acceleration")
        else:
            t, x, y = _columns(args.file, args.time_col, args.x_col, args.y_col)
            params = kalman.KalmanParams(order=args.order, process_var=args.process_var,
                                         measurement_std=args.measurement_std)
            k = kalman.positions_to_kinematics(t, x, y, dt=args.dt, kalman=params,
                                               prefilter_hz=args.prefilter_hz or None)
            v, a = k.profile_points()
        print(insitu.fit_in_situ(v, a, method=args.method).summary())


if __name__ == "__main__":
    main()
