import pytest

import sprintfv as s

pytest.importorskip("weasyprint")
pytest.importorskip("jinja2")


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_sprint_report_is_a_pdf(tmp_path, lang):
    from sprintfv.report import sprint_report

    m = s.load_jb_morin()
    fit = s.fit_radar(m["time"], m["velocity"])
    fv = s.force_velocity_profile(fit.mss, fit.mac, 75, body_height=1.72)
    out = sprint_report(fit, fv, tmp_path / "r.pdf", athlete="Test", optimal=s.optimal_profile(fv, 30),
                        trace=(m["time"], m["velocity"]), lang=lang)
    data = out.read_bytes()
    assert data[:5] == b"%PDF-" and len(data) > 20_000


def test_report_from_split_times(tmp_path):
    from sprintfv.report import sprint_report

    k = s.load_split_times("Kimberley")
    fit = s.fit_splits(k["distance"], k["time"])
    fv = s.force_velocity_profile(fit.mss, fit.mac, 60, body_height=1.7)
    assert sprint_report(fit, fv, tmp_path / "r.pdf").exists()
