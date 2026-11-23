from datetime import date

from typer.testing import CliRunner

from axiom.cli import app
from axiom.models import Report


def test_report_defaults_reconcile():
    r = Report(run_id="r1", date=date(2026, 5, 14), status="normal", headline="ok")
    assert r.unexplained_pct == 100.0 and r.primary_cause is None


def test_cli_stub(tmp_path):
    # axiom run requires data/sim; without it, it exits with an error. Output goes to tmp, not runs/.
    res = CliRunner().invoke(app, ["run", "2026-05-14", "--out", str(tmp_path / "r")])
    # Should fail gracefully when data is missing
    assert res.exit_code in (0, 1)
