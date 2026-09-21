"""Steady-state latency of one any2jev checkpoint: p50 / p95 over repeated single requests.

    python examples/bench_latency.py runs/qwen3-0.6b-public --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

REQUEST = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. "
             "What are you going to do about this? I need an answer today.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                    "shipping": "Delivery status, delays, lost packages",
                                    "billing": "Charges, invoices, payment problems"}},
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--merge", action="store_true", help="merge LoRA into the backbone before measuring")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--questions", type=int, default=3, help="repeat the 3 questions to reach this many")
    ap.add_argument("--state-repeat", type=int, default=1, help="repeat the state text to lengthen it")
    a = ap.parse_args()
    if min(a.n, a.questions, a.state_repeat) < 1:
        ap.error("--n, --questions and --state-repeat must be positive")

    import torch

    from any2jev.model import DecisionModel

    model = DecisionModel.load(a.model_dir, dtype=a.dtype, merge=a.merge)
    req = json.loads(json.dumps(REQUEST))
    req["state"] = " ".join([req["state"]] * a.state_repeat)
    base_qs = list(req["questions"].items())
    req["questions"] = {f"{k}_{i}": v for i in range(a.questions // 3 + 1) for k, v in base_qs}
    req["questions"] = dict(list(req["questions"].items())[: a.questions])
    for _ in range(5):
        model.decide(req)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(a.n):
        t0 = time.perf_counter()
        _, n_in = model.decide(req)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(json.dumps({"model": a.model_dir, "base": model.meta.get("base"), "dtype": str(model.compute_dtype),
                      "device": str(model.device), "mode": model.mode, "input_tokens": n_in,
                      "merged": a.merge, "samples": a.n,
                      "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if model.device.type == "cuda" else None,
                      "questions": len(req["questions"]), "p50_ms": round(statistics.median(times), 1),
                      "p95_ms": round(times[math.ceil(0.95 * len(times)) - 1], 1), "min_ms": round(times[0], 1)}, indent=2))


if __name__ == "__main__":
    main()
