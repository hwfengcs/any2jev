"""Zero-shot baseline: read next-token logits of the *unconverted* base model over lettered options.

This is what "prompt an LLM and look at the label logits" gives you (the SemIf / Nimble-style
inference trick, no training). Reported on the same test JSONL as `any2jev eval`, so the README table
compares like with like: same base weights, same questions, before and after conversion.

    python examples/baseline_zeroshot.py Qwen/Qwen3-0.6B --data data/public/test.jsonl --val data/public/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import string
import time

import numpy as np
import torch

from any2jev.calibration import fit_temperature, summarize
from any2jev.data import load_jsonl, materialize

LETTERS = string.ascii_uppercase


def prompt_for(state: str, spec) -> str:
    lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(spec.options)]
    return (f"State:\n{state}\n\nQuestion: {spec.instructions}\nOptions:\n" + "\n".join(lines) +
            "\n\nAnswer with the letter of the best option.\nAnswer:")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None, help="fit a temperature on (a slice of) this file for a '+T' row")
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]).to(a.device).eval()
    letter_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in LETTERS]

    def run(records):
        logits, labels, qtypes = [], [], []
        t0 = time.time()
        for n, rec in enumerate(records, 1):
            state, specs = materialize(rec)
            for spec in specs:
                ids = tok(prompt_for(state, spec), return_tensors="pt", truncation=True, max_length=a.max_tokens).input_ids.to(a.device)
                z = model(input_ids=ids).logits[0, -1].float()
                logits.append(z[letter_ids[: spec.n_options]].cpu().numpy())
                labels.append(spec.label)
                qtypes.append(spec.qtype)
            if n % 50 == 0:
                print(f"  {n}/{len(records)} records, {len(labels)} questions, {time.time() - t0:.0f}s", flush=True)
        return logits, labels, qtypes

    t = 1.0
    if a.val:
        vl, vy, _ = run(load_jsonl(a.val)[: a.val_n])
        t = fit_temperature(vl, vy)
        print(f"zero-shot temperature fitted on {len(vy)} val questions: T={t:.3f}")
    logits, labels, qtypes = run(load_jsonl(a.data))
    report = {"base": a.base, "n_questions": len(labels), "temperature": t, "metrics": {}, "metrics_T": {}}
    groups = {"overall": list(range(len(labels)))}
    for q in ("noul", "choice", "score"):
        idx = [i for i, x in enumerate(qtypes) if x == q]
        if idx:
            groups[q] = idx
    for g, idx in groups.items():
        report["metrics"][g] = summarize([logits[i] for i in idx], [labels[i] for i in idx], 1.0)
        report["metrics_T"][g] = summarize([logits[i] for i in idx], [labels[i] for i in idx], t)
    print(f"\nzero-shot logit reading, {a.base}, {len(labels)} questions")
    print(f"{'group':8s} {'n':>5s} {'acc':>6s} {'nll':>6s} {'brier':>6s} {'ECE':>6s} | +T: {'nll':>6s} {'brier':>6s} {'ECE':>6s}")
    for g in groups:
        m, mt = report["metrics"][g], report["metrics_T"][g]
        print(f"{g:8s} {m['n']:5d} {m['accuracy']:6.3f} {m['nll']:6.3f} {m['brier']:6.3f} {m['ece']:6.3f} |     {mt['nll']:6.3f} {mt['brier']:6.3f} {mt['ece']:6.3f}")
    if a.out:
        json.dump(report, open(a.out, "w", encoding="utf-8"), indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
