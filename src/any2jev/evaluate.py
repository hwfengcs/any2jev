"""Evaluation: accuracy and calibration per question type, option-order sensitivity, and the
packed-vs-separate isolation check that the block-causal mask must satisfy."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .calibration import summarize
from .data import Record, load_jsonl, materialize
from .model import DecisionModel
from .train import collect_logits


def metrics_by_type(logits, labels, qtypes, temperature: float) -> dict:
    out = {"overall": summarize(logits, labels, temperature)}
    for t in ("noul", "choice", "score"):
        idx = [i for i, q in enumerate(qtypes) if q == t]
        if idx:
            out[t] = summarize([logits[i] for i in idx], [labels[i] for i in idx], temperature)
    return out


@torch.no_grad()
def permutation_sensitivity(model: DecisionModel, records: list[Record], n_perm: int = 4, max_records: int = 50,
                            seed: int = 0, temperature: float | None = None) -> dict:
    """Re-ask each Choice question (>= 2 options) under ``n_perm`` option orders. Reports how often the
    argmax survives reordering and the mean/max spread of any option's probability across orders."""
    rng = random.Random(seed)
    stable, spreads, n = 0, [], 0
    for rec in records[:max_records]:
        state, specs = materialize(rec)
        for spec in specs:
            if spec.qtype != "choice" or spec.n_options < 2:
                continue
            runs = []
            for i in range(n_perm):
                perm = list(range(spec.n_options))
                if i:
                    rng.shuffle(perm)
                p = model.probs([model.encode(state, [spec.permuted(perm)])], temperature=temperature)[0][0].numpy()
                orig = np.zeros(spec.n_options)
                for pos, j in enumerate(perm):
                    orig[j] = p[pos]
                runs.append(orig)
            runs = np.stack(runs)
            stable += int(len(set(runs.argmax(1).tolist())) == 1)
            spreads.append(float((runs.max(0) - runs.min(0)).max()))
            n += 1
    if n == 0:
        return {"n": 0}
    return {"n": n, "argmax_stable_rate": stable / n, "mean_max_spread": float(np.mean(spreads)),
            "p95_max_spread": float(np.percentile(spreads, 95))}


@torch.no_grad()
def isolation_check(model: DecisionModel, records: list[Record], max_records: int = 20,
                    temperature: float | None = None) -> dict:
    """Packed answers must equal answers obtained by asking each question alone (question isolation)."""
    worst = 0.0
    n = 0
    for rec in records[:max_records]:
        state, specs = materialize(rec)
        if len(specs) < 2:
            continue
        packed = model.probs([model.encode(state, specs)], temperature=temperature)[0]
        for spec, p in zip(specs, packed):
            alone = model.probs([model.encode(state, [spec])], temperature=temperature)[0][0]
            worst = max(worst, float((alone - p).abs().max()))
        n += 1
    return {"n": n, "max_abs_prob_diff": worst if n else None}


def evaluate(model: DecisionModel, records: list[Record], *, batch_size: int = 8, n_perm: int = 4,
             perm_records: int = 50, isolation_records: int = 20, temperature: float | None = None) -> dict:
    if not records:
        raise ValueError("evaluation data is empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    t = model.temperature if temperature is None else temperature
    logits, labels, qtypes = collect_logits(model, records, batch_size)
    report = {"n_records": len(records), "n_questions": len(labels), "temperature": t,
              "metrics": metrics_by_type(logits, labels, qtypes, t)}
    if t != 1.0:
        report["metrics_uncalibrated"] = metrics_by_type(logits, labels, qtypes, 1.0)
    if n_perm > 1:
        report["permutation"] = permutation_sensitivity(model, records, n_perm, perm_records, temperature=t)
    if isolation_records > 0:
        report["isolation"] = isolation_check(model, records, isolation_records, temperature=t)
    return report


def evaluate_checkpoint(model_dir: str | Path, data_path: str | Path, out_path: str | Path | None = None,
                        device: str | None = None, dtype: str | None = None, **kw) -> dict:
    model = DecisionModel.load(model_dir, device=device, dtype=dtype)
    report = evaluate(model, load_jsonl(data_path), **kw)
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
