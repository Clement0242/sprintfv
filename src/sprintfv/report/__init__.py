"""Designed PDF report of a sprint force-velocity profile.

    from sprintfv.report import sprint_report
    sprint_report(fit, fv, "report.pdf", athlete="A. Martin", test="Radar, 30 m",
                  optimal=s.optimal_profile(fv, 30), trace=(time, velocity))

Needs the optional dependencies: ``pip install "sprintfv[report]"``.
Branding is the user's: pass ``organisation=`` and ``logo=`` to put your own
name and logo on the report; nothing is added by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import __version__
from ..fit import SprintFit
from ..fvp import ForceVelocityProfile, OptimalProfile, force_velocity_from_trace
from .engine import PALETTE, fig_to_uri, fmt, mpl_theme, render_pdf

TEMPLATES = Path(__file__).resolve().parent / "templates"

LABELS = {
    "fr": {
        "kicker": "Sprint · profil force-vitesse", "title": "Profil <em>force-vitesse</em>",
        "summary": "Synthèse", "reading": "Lecture", "sfv": "Pente F-v", "sprint": "Le sprint",
        "model": "Modèle mono-exponentiel", "param": "Paramètre", "value": "Valeur",
        "mss": "vitesse maximale", "mac": "accélération maximale", "time_correction": "Correction de temps (s)",
        "flying_distance": "Départ lancé (m)", "splits": "Temps de passage modélisés", "distance": "Distance",
        "time": "Temps", "speed": "Vitesse", "force": "Force, puissance, orientation",
        "force_intro": "La force horizontale nette diminue presque linéairement avec la vitesse : son ordonnée à "
                       "l'origine est F0, son intersection avec l'axe des vitesses V0, et la puissance maximale vaut "
                       "F0·V0/4. Le ratio de force (RF) mesure la part de la force totale orientée vers l'avant.",
        "optimal": "Profil optimal", "imbalance": "Déséquilibre FV", "gain": "Gain potentiel", "over": "sur",
        "actual": "Actuel", "optimal_short": "Optimal", "method": "Méthode",
        "cap_sprint": "Vitesse mesurée (points) et modèle ajusté (trait).",
        "cap_fv": "Force horizontale (bleu) et puissance (or) selon la vitesse ; droite F-v en pointillés.",
        "cap_rf": "Ratio de force selon la vitesse : sa pente est l'indice DRF.",
        "cap_opt": "Profil actuel et profil optimal à puissance maximale égale.",
        "balanced": "équilibré", "force_oriented": "orienté force", "velocity_oriented": "orienté vitesse",
        "moderate": "modérément", "marked": "nettement",
    },
    "en": {
        "kicker": "Sprint · force-velocity profile", "title": "Force-velocity <em>profile</em>",
        "summary": "Summary", "reading": "Reading", "sfv": "F-v slope", "sprint": "The sprint",
        "model": "Mono-exponential model", "param": "Parameter", "value": "Value",
        "mss": "maximal speed", "mac": "maximal acceleration", "time_correction": "Time correction (s)",
        "flying_distance": "Flying start (m)", "splits": "Modelled split times", "distance": "Distance",
        "time": "Time", "speed": "Speed", "force": "Force, power, orientation",
        "force_intro": "Net horizontal force decreases almost linearly with velocity: its intercept is F0, its "
                       "velocity intercept V0, and maximal power is F0·V0/4. The ratio of forces (RF) is the share "
                       "of the total force oriented forward.",
        "optimal": "Optimal profile", "imbalance": "FV imbalance", "gain": "Potential gain", "over": "over",
        "actual": "Actual", "optimal_short": "Optimal", "method": "Method",
        "cap_sprint": "Measured speed (dots) and fitted model (line).",
        "cap_fv": "Horizontal force (blue) and power (gold) against velocity; F-v line dashed.",
        "cap_rf": "Ratio of forces against velocity: its slope is the DRF index.",
        "cap_opt": "Actual and optimal profiles at the same maximal power.",
        "balanced": "balanced", "force_oriented": "force-oriented", "velocity_oriented": "velocity-oriented",
        "moderate": "moderately", "marked": "markedly",
    },
}

REFERENCES = [
    "Samozino P. et al. (2016). A simple method for measuring power, force, velocity properties, and mechanical "
    "effectiveness in sprint running. Scand J Med Sci Sports 26, 648–658.",
    "Samozino P. et al. (2022). Optimal mechanical force-velocity profile for sprint acceleration performance. "
    "Scand J Med Sci Sports 32, 559–575.",
    "Morin J-B. & Samozino P. (2016). Interpreting power-force-velocity profiles for individualized and specific "
    "training. Int J Sports Physiol Perform 11, 267–272.",
    "Arsac L. & Locatelli E. (2002). Modeling the energetics of 100-m running by using speed curves of world "
    "champions. J Appl Physiol 92, 1781–1788.",
]


def _imbalance_label(x: float, L: dict, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    if abs(x) <= low:
        return L["balanced"]
    side = L["force_oriented"] if x > 0 else L["velocity_oriented"]
    return f'{L["moderate"] if abs(x) <= high else L["marked"]} {side}'


def _figures(fit: SprintFit, fv: ForceVelocityProfile, opt: OptimalProfile | None, trace, lang: str) -> dict:
    import matplotlib.pyplot as plt

    mpl_theme()
    P = PALETTE
    out = {}
    # 1. The sprint
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    if fit.method == "radar" and fit.predictor.size:
        tc = fit.corrections.get("time_correction", 0.0)
        ax.plot(fit.predictor, fit.observed, ".", color=P["gold"], ms=3, alpha=.8)
        tt = np.linspace(max(0, tc), fit.predictor.max(), 300)
        ax.plot(tt, fit.velocity_at_time(tt - tc), color=P["navy"], lw=2)
        ax.set(xlabel="t (s)", ylabel="v (m/s)")
    else:
        d = np.linspace(0, max(40, fit.predictor.max() if fit.predictor.size else 40), 200)
        ax.plot(fit.time_at_distance(d), d, color=P["navy"], lw=2)
        if fit.predictor.size:
            ax.plot(fit.observed, fit.predictor, "o", color=P["red"], ms=5)
        ax.set(xlabel="t (s)", ylabel="distance (m)")
    out["fig_sprint"] = fig_to_uri(fig)

    # 2. Force and power against velocity
    tab = fv.table
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    vv = np.linspace(0, fv.v0, 100)
    if tab:
        ax.plot(tab["velocity"], tab["force"] / fv.body_mass, color=P["navy"], lw=2.2 if fv.method != "direct" else 0,
                marker="" if fv.method != "direct" else ".", ms=3)
    ax.plot(vv, fv.f0_rel * (1 - vv / fv.v0), "--", color=P["navy"], lw=1.1)
    ax.set(xlabel="v (m/s)", ylabel="F (N/kg)", xlim=(0, fv.v0 * 1.03), ylim=(0, fv.f0_rel * 1.12))
    ax2 = ax.twinx()
    pv = fv.f0_rel * vv * (1 - vv / fv.v0)
    ax2.plot(vv, pv, color=P["gold"], lw=2.2)
    ax2.set_ylabel("P (W/kg)", color=P["gold_dark"])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, pv.max() * 1.25)
    out["fig_fv"] = fig_to_uri(fig)

    # 3. Ratio of forces
    if tab and fv.rf_max is not None:
        m = (tab["time"] - tab["time"][0]) >= 0.3
        fig, ax = plt.subplots(figsize=(7.4, 1.8))
        ax.plot(tab["velocity"][m], tab["rf"][m], color=P["red"], lw=2 if fv.method != "direct" else 0,
                marker="" if fv.method != "direct" else ".", ms=3)
        b = np.polyfit(tab["velocity"][m], tab["rf"][m], 1)
        xv = np.linspace(tab["velocity"][m].min(), tab["velocity"][m].max(), 10)
        ax.plot(xv, np.polyval(b, xv), "--", color=P["navy"], lw=1.1)
        ax.set(xlabel="v (m/s)", ylabel="RF (%)")
        out["fig_rf"] = fig_to_uri(fig)

    # 4. Optimal profile
    if opt is not None:
        fig, ax = plt.subplots(figsize=(4.0, 2.7))
        vmax = max(fv.v0, opt.v0_opt) * 1.05
        v1 = np.linspace(0, fv.v0, 50)
        v2 = np.linspace(0, opt.v0_opt, 50)
        ax.plot(v1, fv.f0_rel * (1 - v1 / fv.v0), color=P["navy"], lw=2.2, label="actuel" if lang == "fr" else "actual")
        ax.plot(v2, opt.f0_opt_rel * (1 - v2 / opt.v0_opt), color=P["red"], lw=2.2, ls="--",
                label="optimal")
        ax.set(xlabel="v (m/s)", ylabel="F (N/kg)", xlim=(0, vmax), ylim=(0, max(fv.f0_rel, opt.f0_opt_rel) * 1.12))
        ax.legend()
        out["fig_opt"] = fig_to_uri(fig)
    return out


def _reading(fit, fv, opt, trace, lang, L, thresholds, body_mass, body_height) -> list[str]:
    f = lambda x, d=2: fmt(x, d, lang)
    lines = []
    if lang == "fr":
        lines.append(f"Force maximale théorique <b>F0 = {f(fv.f0_rel)} N/kg</b>, vitesse maximale théorique "
                     f"<b>V0 = {f(fv.v0)} m/s</b>, puissance maximale <b>{f(fv.pmax_rel, 1)} W/kg</b>.")
        if opt is not None:
            lines.append(f"Comparé au profil optimal sur {f(opt.distance, 0)} m, le profil est "
                         f"<b>{_imbalance_label(opt.fv_imbalance, L, thresholds)}</b> (déséquilibre "
                         f"{f(opt.fv_imbalance, 0)} %) ; à puissance égale, le profil optimal gagnerait "
                         f"{f(opt.time_gain, 3)} s.")
        if fit.observed.size:
            lines.append(f"Qualité de l'ajustement du modèle : R² = {f(fit.r2, 4)}.")
    else:
        lines.append(f"Theoretical maximal force <b>F0 = {f(fv.f0_rel)} N/kg</b>, theoretical maximal velocity "
                     f"<b>V0 = {f(fv.v0)} m/s</b>, maximal power <b>{f(fv.pmax_rel, 1)} W/kg</b>.")
        if opt is not None:
            lines.append(f"Compared with the optimal profile over {f(opt.distance, 0)} m, the profile is "
                         f"<b>{_imbalance_label(opt.fv_imbalance, L, thresholds)}</b> (imbalance "
                         f"{f(opt.fv_imbalance, 0)} %); at equal power the optimal profile would save "
                         f"{f(opt.time_gain, 3)} s.")
        if fit.observed.size:
            lines.append(f"Goodness of fit of the model: R² = {f(fit.r2, 4)}.")
    if trace is not None and fv.method != "direct":
        try:
            d = force_velocity_from_trace(*trace, body_mass, body_height=body_height)
            lines.append(
                (f"Calculée directement sur la vitesse mesurée, sans modèle, F0 vaut {f(d.f0_rel)} N/kg et V0 "
                 f"{f(d.v0)} m/s (linéarité R² = {f(d.r2)})." if lang == "fr" else
                 f"Computed directly on the measured speed, without the model, F0 is {f(d.f0_rel)} N/kg and V0 "
                 f"{f(d.v0)} m/s (linearity R² = {f(d.r2)})."))
        except ValueError:
            pass
    return lines


def _method(fit, fv, lang, thresholds) -> list[str]:
    lo, hi = thresholds
    if lang == "fr":
        src = {"radar": "une mesure de vitesse continue (radar, laser ou système de positionnement)",
               "splits": "des temps de passage"}.get(fit.method, fit.method)
        p = [f"Le modèle mono-exponentiel v(t) = MSS·(1 − e<sup>−t/τ</sup>) est ajusté par moindres carrés sur "
             f"{src}. La force horizontale nette est F<sub>H</sub> = m·a + k·v², avec un coefficient aérodynamique "
             f"k calculé d'après la masse et la taille (Arsac &amp; Locatelli 2002).",
             "F0, V0 et Pmax proviennent de la régression linéaire de F<sub>H</sub> sur v (Samozino et al. 2016). "
             "Le ratio de force RF = F<sub>H</sub>/√(F<sub>H</sub>² + (m·g)²) est calculé à partir de 0,3 s ; DRF est "
             "la pente de RF selon la vitesse." if fv.method != "direct" else
             "Voie directe : F<sub>H</sub> est calculée sur la vitesse mesurée et sa dérivée, sans imposer de forme "
             "exponentielle ; le R² de la régression teste la linéarité du profil.",
             f"Le profil optimal est celui qui minimise le temps sur la distance à puissance maximale égale, obtenu "
             f"par simulation numérique (Samozino et al. 2022). Seuils de lecture du déséquilibre : ±{lo:g} % "
             f"équilibré, au-delà de ±{hi:g} % nettement orienté (indicatifs, configurables)."]
    else:
        src = {"radar": "a continuous speed measurement (radar, laser or positioning system)",
               "splits": "split times"}.get(fit.method, fit.method)
        p = [f"The mono-exponential model v(t) = MSS·(1 − e<sup>−t/τ</sup>) is fitted by least squares to {src}. "
             f"Net horizontal force is F<sub>H</sub> = m·a + k·v², with an aerodynamic coefficient k from body mass "
             f"and height (Arsac &amp; Locatelli 2002).",
             "F0, V0 and Pmax come from the linear regression of F<sub>H</sub> on v (Samozino et al. 2016). The ratio "
             "of forces RF = F<sub>H</sub>/√(F<sub>H</sub>² + (m·g)²) is computed from 0.3 s; DRF is the slope of RF "
             "against velocity." if fv.method != "direct" else
             "Direct route: F<sub>H</sub> is computed on the measured speed and its derivative, with no exponential "
             "shape imposed; the R² of the regression tests linearity.",
             f"The optimal profile minimises the time over the distance at equal maximal power, by numerical "
             f"simulation (Samozino et al. 2022). Reading thresholds for the imbalance: ±{lo:g} % balanced, beyond "
             f"±{hi:g} % markedly oriented (indicative, configurable)."]
    return p


def sprint_report(
    fit: SprintFit,
    fv: ForceVelocityProfile,
    out: str | Path,
    *,
    athlete: str | None = None,
    test: str | None = None,
    test_date: str | None = None,
    optimal: OptimalProfile | None = None,
    trace: tuple[np.ndarray, np.ndarray] | None = None,
    body_height: float = 1.75,
    lang: str = "fr",
    organisation: str | None = None,
    logo: str | Path | None = None,
    imbalance_thresholds: tuple[float, float] = (10.0, 25.0),
    split_distances: tuple[float, ...] = (5, 10, 20, 30, 40),
) -> Path:
    """Write a designed PDF report of a sprint profile and return its path.

    ``trace`` (time, velocity) adds the direct-route comparison. ``logo`` and
    ``organisation`` put your own branding on the report.
    """
    if lang not in LABELS:
        raise ValueError("lang must be 'fr' or 'en'")
    L = LABELS[lang]
    d = np.asarray(split_distances, dtype=float)
    splits = list(zip(d, fit.time_at_distance(d), fit.velocity_at_time(fit.time_at_distance(d))))
    ctx = {
        "lang": lang, "L": L, "fit": fit, "fv": fv, "opt": optimal, "splits": splits,
        "title": "Profil force-vitesse" if lang == "fr" else "Force-velocity profile",
        "title_html": L["title"], "kicker": L["kicker"], "athlete": athlete, "test": test, "test_date": test_date,
        "extra_meta": {("Masse" if lang == "fr" else "Body mass"): f"{fmt(fv.body_mass, 1, lang)} kg"},
        "organisation": organisation, "logo": logo, "tool": "sprintfv", "version": __version__,
        "imbalance_label": _imbalance_label(optimal.fv_imbalance, L, imbalance_thresholds) if optimal else "",
        "reading": _reading(fit, fv, optimal, trace, lang, L, imbalance_thresholds, fv.body_mass, body_height),
        "method": _method(fit, fv, lang, imbalance_thresholds), "references": REFERENCES,
        "disclaimer": ("Les valeurs dépendent de la qualité de la mesure ; elles s'interprètent avec le contexte de "
                       "l'athlète." if lang == "fr" else
                       "Values depend on measurement quality; read them with the athlete's context in mind."),
        **_figures(fit, fv, optimal, trace, lang),
    }
    return render_pdf(TEMPLATES, "sprint.html", ctx, out)


__all__ = ["sprint_report"]
