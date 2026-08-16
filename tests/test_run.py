"""Orchestrator writes metrics.json for the cantilever fixture."""

from pathlib import Path

from armpipe.run import run

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cantilever.yaml"


def test_run_cantilever_writes_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = run(FIXTURE, tmp_path / "work")
    assert path.is_file()
    data = __import__("json").loads(path.read_text())
    assert "max_von_mises_MPa" in data
    assert "tip_deflection_mm" in data
    assert data["tip_deflection_mm"] == __import__("pytest").approx(1.1429, rel=0.05)
    assert (tmp_path / "metrics.json").is_file()
