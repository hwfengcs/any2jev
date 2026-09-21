"""Measure a hosted LLM with structured outputs on the same test JSONL, so the README table can carry
a real number instead of a guess. Works with any OpenAI-compatible endpoint (OpenAI, a relay, vLLM).

    export OPENAI_API_KEY=...            # never read from anywhere else
    python examples/bench_cloud.py --model gpt-4o --data data/public/test.jsonl --n 200 \
        --out runs/qwen3-0.6b-public/baseline_cloud_gpt-4o.json

Reports per-question latency (p50/p95), format failure rate and accuracy. Costs whatever your provider
charges for ~300 input + ~15 output tokens per question; it prints the token totals so you can price it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

from baseline_generate_json import (  # same prompt and parser as the local baseline
    build_messages,
    parse_answer,
)

from any2jev.data import load_jsonl, materialize


def call(base_url: str, key: str, model: str, messages: list[dict], timeout: float) -> tuple[str, dict]:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 32,
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="records to sample from the start of the file")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) first")
    rows, usage_in, usage_out = [], 0, 0
    for i, rec in enumerate(load_jsonl(a.data)[: a.n], 1):
        state, specs = materialize(rec)
        for spec in specs:
            t0 = time.perf_counter()
            try:
                text, usage = call(a.base_url, key, a.model, build_messages(state, spec), a.timeout)
                err = None
            except Exception as e:  # noqa: BLE001
                text, usage, err = "", {}, repr(e)
            ms = (time.perf_counter() - t0) * 1000
            usage_in += usage.get("prompt_tokens", 0)
            usage_out += usage.get("completion_tokens", 0)
            idx = parse_answer(text, spec) if not err else None
            rows.append({"qtype": spec.qtype, "format_ok": idx is not None, "correct": idx == spec.label,
                         "latency_ms": round(ms, 1), "error": err})
        if i % 20 == 0:
            print(f"  {i}/{a.n} records, acc {statistics.mean(r['correct'] for r in rows):.3f}, "
                  f"p50 {sorted(r['latency_ms'] for r in rows)[len(rows) // 2]:.0f} ms", flush=True)
    lat = sorted(r["latency_ms"] for r in rows)
    m = {"n": len(rows), "format_failure_rate": 1 - statistics.mean(r["format_ok"] for r in rows),
         "accuracy": statistics.mean(r["correct"] for r in rows), "latency_p50_ms": lat[len(lat) // 2],
         "latency_p95_ms": lat[int(0.95 * len(lat)) - 1], "errors": sum(1 for r in rows if r["error"])}
    report = {"model": a.model, "base_url": a.base_url, "method": "hosted LLM, JSON mode (response_format=json_object)",
              "metrics": {"overall": m}, "tokens": {"input": usage_in, "output": usage_out}}
    for t in ("noul", "choice", "score"):
        sub = [r for r in rows if r["qtype"] == t]
        if sub:
            lat_t = sorted(r["latency_ms"] for r in sub)
            report["metrics"][t] = {"n": len(sub), "format_failure_rate": 1 - statistics.mean(r["format_ok"] for r in sub),
                                    "accuracy": statistics.mean(r["correct"] for r in sub), "latency_p50_ms": lat_t[len(lat_t) // 2]}
    json.dump(report, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"\n{a.model}: n={m['n']} acc {m['accuracy']:.3f} format failures {m['format_failure_rate']:.1%} "
          f"p50 {m['latency_p50_ms']:.0f} ms p95 {m['latency_p95_ms']:.0f} ms | tokens in {usage_in} out {usage_out}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
