"""Regenerate the README figures:  python examples/make_figures.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sprintfv as s

OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "figure.dpi": 150})
BLUE, ORANGE, GREY = "#2f6fdc", "#e8743b", "#c8c8c8"

# 1. Radar sprint -> model -> force-velocity profile -----------------------------
m = s.load_jb_morin()
f = s.fit_radar(m["time"], m["velocity"])
fv = s.force_velocity_profile(f.mss, f.mac, body_mass=75, body_height=1.72)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
a1.plot(m["time"], m["velocity"], ".", color=GREY, ms=4, label="radar")
tt = np.linspace(0, m["time"].max(), 300)
a1.plot(tt, f.velocity_at_time(tt - f.corrections["time_correction"]), color=BLUE, lw=2,
        label=f"model: MSS {f.mss:.2f} m/s, MAC {f.mac:.2f} m/s²")
a1.set(xlabel="time (s)", ylabel="velocity (m/s)", title="1. Fit the sprint")
a1.legend(loc="lower right", frameon=False)
tab = fv.table
a2.plot(tab["velocity"], tab["force"] / fv.body_mass, color=BLUE, lw=2, label="horizontal force")
vv = np.linspace(0, fv.v0, 50)
a2.plot(vv, fv.f0_rel * (1 - vv / fv.v0), "--", color="k", lw=1, label="linear F-V profile")
a2.set(xlabel="velocity (m/s)", ylabel="force (N/kg)", ylim=(0, None), xlim=(0, fv.v0 * 1.02),
       title=f"2. F0 {fv.f0_rel:.2f} N/kg · V0 {fv.v0:.2f} m/s · Pmax {fv.pmax_rel:.1f} W/kg")
a2b = a2.twinx()
a2b.plot(tab["velocity"], tab["power"] / fv.body_mass, color=ORANGE, lw=1.5)
a2b.set_ylabel("power (W/kg)", color=ORANGE)
a2b.grid(False)
a2.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "radar_fvp.png")

# 2. Tracking positions -> Kalman -> four in-situ methods ------------------------
d = s.load_lps_session()
k = s.positions_to_kinematics(d["time"], d["x"], d["y"], bounds=(-2, 32, -2, 17))
v, a = k.profile_points()
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
names = {"linear": "Linear (Morin 2021)", "hyperbolic": "Hyperbolic (Fenaux)",
         "tukey": "Tukey fences", "prediction_interval": "Prediction interval"}
for ax, method in zip(axes, s.IN_SITU_METHODS):
    p = s.fit_in_situ(v, a, method=method)
    ax.plot(v[::4], a[::4], ".", color=GREY, ms=1, alpha=0.5)
    ax.plot(p.speed, p.acceleration, ".", color=BLUE, ms=2 if method == "hyperbolic" else 5)
    xs = np.linspace(0, p.s0, 200)
    ax.plot(xs, p.curve(xs), color=ORANGE, lw=2)
    ax.set(title=f"{names[method]}\nA0 {p.a0:.1f} · peak {p.peak_acceleration:.1f} · S0 {p.s0:.1f}",
           xlabel="speed (m/s)", xlim=(0, 11), ylim=(0, 11))
axes[0].set_ylabel("acceleration (m/s²)")
fig.suptitle("One 95-min basketball practice (LPS, 20 Hz): positions → Kalman + RTS → in-situ profile", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "in_situ_methods.png", bbox_inches="tight")

# 3. Effect of the Kalman process noise -------------------------------------------
t_peak = k.time[np.nanargmax(np.where(k.usable, k.acceleration, np.nan))]
sel = (k.time > t_peak - 4) & (k.time < t_peak + 4)
fig, ax = plt.subplots(figsize=(10, 3.2))
for pv, col in [(0.5, "#9bbcf0"), (5, BLUE), (50, ORANGE)]:
    kk = s.positions_to_kinematics(d["time"], d["x"], d["y"], bounds=(-2, 32, -2, 17),
                                   kalman=s.KalmanParams(process_var=pv))
    ax.plot(kk.time[sel], kk.acceleration[sel], color=col, lw=1.4, label=f"process_var = {pv:g}")
ax.set(xlabel="time (s)", ylabel="acceleration (m/s²)",
       title="KalmanParams(process_var=...): how closely the estimate follows the data")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(OUT / "kalman_process_var.png")
print("figures written to", OUT)
