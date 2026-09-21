import json

import pytest
import torch

from any2jev.data import (
    load_jsonl,
    materialize,
    save_jsonl,
    shuffle_choices,
    split_records,
    synthetic_records,
)
from any2jev.evaluate import evaluate_checkpoint
from any2jev.train import TrainConfig, train


def test_synthetic_records_are_valid_and_varied():
    recs = synthetic_records(200, seed=3)
    types = set()
    for r in recs:
        state, specs = materialize(r)
        assert state and 1 <= len(specs) <= 4
        for s in specs:
            assert s.label is not None and 0 <= s.label < s.n_options
            types.add(s.qtype)
    assert types == {"noul", "choice", "score"}
    assert any(isinstance(r["state"], dict) for r in recs) and any(isinstance(r["state"], str) for r in recs)


def test_jsonl_round_trip_and_split(tmp_path):
    recs = synthetic_records(50, seed=1)
    p = tmp_path / "d.jsonl"
    assert save_jsonl(p, recs) == 50
    assert load_jsonl(p) == recs
    train_recs, val_recs = split_records(recs, 0.2, seed=0)
    assert len(train_recs) == 40 and len(val_recs) == 10


def test_shuffle_choices_preserves_labels():
    import random

    rng = random.Random(0)
    for r in synthetic_records(30, seed=2):
        _, specs = materialize(r)
        for a, b in zip(specs, shuffle_choices(specs, rng)):
            assert a.keys[a.label] == b.keys[b.label] and sorted(a.options) == sorted(b.options)


def test_train_eval_smoke(tiny_base, tmp_path):
    recs = synthetic_records(28, seed=5)
    save_jsonl(tmp_path / "train.jsonl", recs[:20])
    save_jsonl(tmp_path / "val.jsonl", recs[20:])
    cfg = TrainConfig(base=tiny_base, data=str(tmp_path / "train.jsonl"), val=str(tmp_path / "val.jsonl"),
                      out=str(tmp_path / "ckpt"), epochs=1, batch_size=4, lora_r=4, head_dim=16, device="cpu",
                      log_every=1, brier_weight=0.5, ordinal_weight=0.5)
    logs = []
    out = train(cfg, log=logs.append)
    rep = json.loads((out / "train_report.json").read_text())
    assert len(rep["history"]) == 5 and all("loss" in h for h in rep["history"])
    assert "temperature" in rep and rep["val_calibrated"]["n"] == rep["val_uncalibrated"]["n"] > 0
    assert json.loads((out / "any2jev.json").read_text())["temperature"] == rep["temperature"]
    # continue training from the checkpoint (load path with trainable adapters)
    cfg2 = TrainConfig(base=str(out), data=cfg.data, out=str(tmp_path / "ckpt2"), max_steps=2, batch_size=4, device="cpu",
                       calibrate=False)
    out2 = train(cfg2, log=logs.append)
    assert json.loads((out2 / "any2jev.json").read_text())["temperature"] == 1.0
    r = evaluate_checkpoint(out, tmp_path / "val.jsonl", tmp_path / "eval.json", device="cpu", n_perm=2, perm_records=5)
    assert r["metrics"]["overall"]["n"] == rep["val_calibrated"]["n"]
    assert r["isolation"]["max_abs_prob_diff"] < 1e-4
    assert r["permutation"]["n"] > 0 and 0 <= r["permutation"]["argmax_stable_rate"] <= 1
    assert (tmp_path / "eval.json").exists()


def test_accumulation_matches_large_batch_with_uneven_questions_and_tail(tiny_base, tmp_path, monkeypatch):
    from any2jev.model import DecisionModel

    # Five records: one full accumulation window and a one-record tail.
    records = synthetic_records(5, seed=4)
    records[0]["questions"] = dict(list(records[0]["questions"].items())[:1])
    data = tmp_path / "train.jsonl"
    save_jsonl(data, records)
    models, seen, gradients = [], [], []
    original_step = torch.optim.AdamW.step

    def step(opt, *args, **kwargs):
        gradients.append([p.grad.detach().clone() for g in opt.param_groups for p in g["params"]])
        return original_step(opt, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)

    def create(cfg):
        model = DecisionModel.from_base(tiny_base, lora_r=4, head_dim=16, lora_dropout=0, device="cpu")
        models.append(model)
        original_encode = model.encode

        def encode(*args, **kwargs):
            seen.append(args[0])
            return original_encode(*args, **kwargs)

        monkeypatch.setattr(model, "encode", encode)
        return model

    monkeypatch.setattr("any2jev.train.load_or_create", create)
    for batch, accum in ((4, 1), (2, 2)):
        seen.clear()
        train(TrainConfig(base=tiny_base, data=str(data), out=str(tmp_path / f"b{batch}"), epochs=1,
                          batch_size=batch, grad_accum=accum, shuffle_options=False, max_grad_norm=0),
              log=lambda _: None)
        assert len(seen) == len(records), "one epoch must consume every record exactly once"
    assert len(gradients) == 4
    for large, accumulated in zip(gradients[:2], gradients[2:]):
        for a, b in zip(large, accumulated):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)
    # Compare observable probabilities too; near-zero gradients of the softmax-invariant
    # key bias can produce different Adam updates without affecting the distribution.
    for model in models:
        model.eval()
    state, specs = materialize(records[0])
    a, b = [m.probs([m.encode(state, specs)])[0] for m in models]
    for p, q in zip(a, b):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"grad_accum": 0}, {"epochs": 0}, {"epochs": float("nan")},
    {"max_steps": 0}, {"log_every": 0}, {"max_state": 0}, {"max_branch": -1},
])
def test_invalid_training_config_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(base="unused", data="unused", out="unused", **kwargs)


def test_empty_training_data_fails_before_model_loading(tmp_path, monkeypatch):
    data = tmp_path / "empty.jsonl"
    data.write_text("\n", encoding="utf-8")

    def no_load(_):
        pytest.fail("empty data must be rejected before loading model weights")

    monkeypatch.setattr("any2jev.train.load_or_create", no_load)
    with pytest.raises(ValueError, match="empty"):
        train(TrainConfig(base="unused", data=str(data), out=str(tmp_path / "out")))
