from typer.testing import CliRunner

from any2jev.cli import app


def test_cli_help_lists_commands():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("convert", "train", "calibrate", "eval", "serve", "ask", "push", "data"):
        assert cmd in r.output


def test_resolve_model_dir_passthrough(tmp_path):
    from any2jev.hub import resolve_model_dir

    assert resolve_model_dir(str(tmp_path)) == str(tmp_path)


def test_cli_data_synthetic(tmp_path):
    r = CliRunner().invoke(app, ["data", "synthetic", "--out", str(tmp_path), "--n", "30"])
    assert r.exit_code == 0, r.output
    for split, n in (("train", 24), ("val", 3), ("test", 3)):
        assert sum(1 for _ in open(tmp_path / f"{split}.jsonl", encoding="utf-8")) == n


def test_eval_preserves_hub_uri(monkeypatch):
    calls = []

    def evaluate(path, *args, **kwargs):
        calls.append(path)
        return {"temperature": 1.0, "n_questions": 0, "metrics": {},
                "isolation": {"n": 0, "max_abs_prob_diff": None}}

    monkeypatch.setattr("any2jev.evaluate.evaluate_checkpoint", evaluate)
    result = CliRunner().invoke(app, ["eval", "hf://user/model@v1", "--data", "test.jsonl"])
    assert result.exit_code == 0, result.output
    assert calls == ["hf://user/model@v1"]
    assert "skipped" in result.output
