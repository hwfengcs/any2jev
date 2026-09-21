from typer.testing import CliRunner

from any2jev.cli import app


def test_cli_help_lists_commands():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("convert", "train", "calibrate", "eval", "serve", "ask", "data"):
        assert cmd in r.output


def test_cli_data_synthetic(tmp_path):
    r = CliRunner().invoke(app, ["data", "synthetic", "--out", str(tmp_path), "--n", "30"])
    assert r.exit_code == 0, r.output
    for split, n in (("train", 24), ("val", 3), ("test", 3)):
        assert sum(1 for _ in open(tmp_path / f"{split}.jsonl", encoding="utf-8")) == n
