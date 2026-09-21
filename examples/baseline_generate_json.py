"""Baseline: the *unconverted* base model, prompted for a one-key JSON answer and decoded with
``generate()``. Measures what any2jev replaces: format failures, accuracy and per-question latency of
autoregressive JSON on the same test JSONL that ``any2jev eval`` uses.

    python examples/baseline_generate_json.py Qwen/Qwen3-0.6B --data data/public/test.jsonl \
        --out runs/qwen3-0.6b-public/baseline_generate_json.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time

import torch

from any2jev.data import load_jsonl, materialize


def build_messages(state: str, spec) -> list[dict]:
    if spec.qtype == "choice":
        allowed = "; ".join(f'"{k}"' + (f" = {o[len(k) + 2:]}" if o.startswith(k + ": ") else "") for k, o in zip(spec.keys, spec.options))
        rule = f"Answer with exactly one of these keys: {allowed}."
        example = '{"answer": "<key>"}'
    elif spec.qtype == "noul":
        crit = [o for o in spec.options if ": " in o]
        rule = "Answer true or false." + (f" ({'; '.join(crit)})" if crit else "")
        example = '{"answer": true}'
    else:
        levels = ", ".join(f"{i} = {o}" for i, o in enumerate(spec.options))
        rule = f"Answer with an integer level: {levels}."
        example = '{"answer": <integer>}'
    content = (f"{state}\n\nQuestion: {spec.instructions}\n{rule}\n"
               f"Reply with ONE JSON object of the form {example} and nothing else.")
    return [{"role": "user", "content": content}]


def parse_answer(text: str, spec) -> int | None:
    """Option index if the reply is a valid JSON object whose ``answer`` is an allowed value, else None."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "answer" not in obj:
        return None
    v = obj["answer"]
    if spec.qtype == "choice":
        if isinstance(v, str):
            s = v.strip().lower()
            for i, (k, o) in enumerate(zip(spec.keys, spec.options)):
                if s in (k.lower(), o.lower()):
                    return i
        return None
    if spec.qtype == "noul":
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, str) and v.strip().lower() in ("true", "false", "yes", "no"):
            return int(v.strip().lower() in ("true", "yes"))
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int) and 0 <= v < spec.n_options:
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s.isdigit() and 0 <= int(s) < spec.n_options:
            return int(s)
        for i, o in enumerate(spec.options):
            if s == o.lower():
                return i
    return None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype).cuda().eval()
    records = load_jsonl(a.data)[: a.max_records]
    rows, t_start = [], time.time()
    for n, rec in enumerate(records, 1):
        state, specs = materialize(rec)
        for spec in specs:
            prompt = tok.apply_chat_template(build_messages(state, spec), add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            gen = out[0, inputs.input_ids.shape[1]:]
            text = tok.decode(gen, skip_special_tokens=True)
            idx = parse_answer(text, spec)
            rows.append({"qtype": spec.qtype, "format_ok": idx is not None, "correct": idx == spec.label,
                         "latency_ms": round(ms, 1), "output_tokens": int(gen.shape[0]), "text": text[:120]})
        if n % 50 == 0:
            done = sum(1 for r in rows)
            print(f"  {n}/{len(records)} records, {done} q, fmt_fail {1 - statistics.mean(r['format_ok'] for r in rows):.3f}, "
                  f"acc {statistics.mean(r['correct'] for r in rows):.3f}, {time.time() - t_start:.0f}s", flush=True)

    def summary(sub):
        lat = sorted(r["latency_ms"] for r in sub)
        valid = [r for r in sub if r["format_ok"]]
        return {"n": len(sub), "format_failure_rate": 1 - len(valid) / len(sub),
                "accuracy": statistics.mean(r["correct"] for r in sub),
                "accuracy_on_valid": statistics.mean(r["correct"] for r in valid) if valid else None,
                "latency_p50_ms": lat[len(lat) // 2], "latency_p95_ms": lat[int(0.95 * len(lat)) - 1],
                "mean_output_tokens": statistics.mean(r["output_tokens"] for r in sub)}

    report = {"base": a.base, "method": "prompted for JSON, transformers.generate (greedy)", "dtype": a.dtype,
              "gpu": torch.cuda.get_device_name(0), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
              "max_new_tokens": a.max_new_tokens, "metrics": {"overall": summary(rows)}, "examples_failed": []}
    for t in ("noul", "choice", "score"):
        sub = [r for r in rows if r["qtype"] == t]
        if sub:
            report["metrics"][t] = summary(sub)
    report["examples_failed"] = [r["text"] for r in rows if not r["format_ok"]][:12]
    json.dump(report, open(a.out, "w", encoding="utf-8"), indent=2)
    m = report["metrics"]["overall"]
    print(f"\n{a.base} prompted for JSON: n={m['n']} format failures {m['format_failure_rate']:.1%} "
          f"acc {m['accuracy']:.3f} (on valid {m['accuracy_on_valid']:.3f}) p50 {m['latency_p50_ms']} ms "
          f"peak VRAM {report['peak_vram_mb']} MB")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
