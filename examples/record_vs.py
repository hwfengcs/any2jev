"""Record real timings for the split-screen comparison: the same base weights answering the same
three questions, (left) prompted for JSON and decoded token by token, (right) converted with any2jev
and answered in one forward pass. Writes ``runs/vs_recording.json`` for ``render_vs.py``.

    python examples/record_vs.py --base Qwen/Qwen3-0.6B --checkpoint runs/qwen3-0.6b-public
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import threading
import time

import torch

NL = chr(10)

STATE = ("Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. "
         "What are you going to do about this? I need an answer today.")
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"}},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}


def json_messages(state: str, questions: dict) -> list[dict]:
    lines = ["Read the customer message and answer every question.",
             "Reply with ONE JSON object and nothing else.", "", f"Message: {state}", "", "Questions:"]
    for qid, q in questions.items():
        if q["type"] == "choice":
            opts = "; ".join(f"{k} = {v}" for k, v in q["criteria"].items())
            lines.append(f'- "{qid}": {q["instructions"]} Answer with exactly one of: {opts}.')
        elif q["type"] == "noul":
            lines.append(f'- "{qid}": {q["instructions"]} Answer true or false.')
        else:
            levels = ", ".join(f"{i} = {lvl}" for i, lvl in enumerate(q["criteria"]))
            lines.append(f'- "{qid}": {q["instructions"]} Answer with an integer level: {levels}.')
    lines += ["", 'Format: {"department": "<team>", "escalate": <true|false>, "frustration": <0|1|2>}']
    return [{"role": "user", "content": "\n".join(lines)}]


def parse_json_answer(text: str, questions: dict) -> tuple[bool, dict | None]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False, None
    if not isinstance(obj, dict):
        return False, None
    for qid, q in questions.items():
        if qid not in obj:
            return False, obj
        v = obj[qid]
        if q["type"] == "choice" and v not in q["criteria"]:
            return False, obj
        if q["type"] == "noul" and not isinstance(v, bool):
            return False, obj
        if q["type"] == "score" and not (isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(q["criteria"])):
            return False, obj
    return True, obj


def record_generate(base: str, dtype: torch.dtype, device: str, runs: int, max_new_tokens: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    tok = AutoTokenizer.from_pretrained(base)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device).eval()
    prompt = tok.apply_chat_template(json_messages(STATE, QUESTIONS), add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(device)

    def one(max_new: int, stream: bool):
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None
        pieces, t0 = [], time.perf_counter()
        kw = dict(**inputs, max_new_tokens=max_new, do_sample=False, streamer=streamer)
        th = threading.Thread(target=lambda: model.generate(**kw))
        th.start()
        if streamer is not None:
            for piece in streamer:
                pieces.append([round((time.perf_counter() - t0) * 1000, 1), piece])
        th.join()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000, pieces

    one(8, False)  # warm-up
    results = [one(max_new_tokens, True) for _ in range(runs)]
    results.sort(key=lambda r: r[0])
    total_ms, pieces = results[len(results) // 2]
    text = "".join(p for _, p in pieces)
    n_tokens = len(tok(text, add_special_tokens=False).input_ids)
    ok, obj = parse_json_answer(text, QUESTIONS)
    out = {"model": base, "method": "prompted for JSON, transformers.generate (greedy)", "dtype": str(dtype),
           "device": device, "prompt_tokens": int(inputs.input_ids.shape[1]), "output_tokens": n_tokens,
           "total_ms": round(total_ms, 1), "all_runs_ms": [round(r[0], 1) for r in results],
           "first_token_ms": pieces[0][0] if pieces else None, "pieces": pieces, "output_text": text,
           "format_ok": ok, "parsed": obj, "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
    return out


def record_logit_reading(base: str, dtype: torch.dtype, device: str, runs: int) -> dict:
    """Zero-shot label-logit reading (SemIf-style): one prefill per question, argmax over letter tokens."""
    import string

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from any2jev.schema import SystemOneRequest, render, to_specs

    tok = AutoTokenizer.from_pretrained(base)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device).eval()
    specs = to_specs(SystemOneRequest.model_validate({"state": STATE, "questions": QUESTIONS}))
    letters = string.ascii_uppercase
    letter_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in letters]
    prompts = []
    for spec in specs:
        opts = NL.join(f"{letters[i]}. {o}" for i, o in enumerate(spec.options))
        text = (f"State:{NL}{render(STATE)}{NL}{NL}Question: {spec.instructions}{NL}Options:{NL}{opts}{NL}{NL}"
                f"Answer with the letter of the best option.{NL}Answer:")
        prompts.append(tok(text, return_tensors="pt").input_ids.to(device))

    @torch.no_grad()
    def one():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = []
        for ids, spec in zip(prompts, specs):
            z = model(input_ids=ids).logits[0, -1]
            outs.append(torch.softmax(z[letter_ids[: spec.n_options]].float(), -1).tolist())
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000, outs

    one()
    times = [one()[0] for _ in range(runs)]
    _, probs = one()
    return {"model": base, "method": "label-logit reading, one prefill per question, no training", "dtype": str(dtype),
            "prompt_tokens": [int(p.shape[1]) for p in prompts], "total_ms": round(statistics.median(times), 1),
            "probs": dict(zip(QUESTIONS, probs)), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}


def record_any2jev(checkpoint: str, dtype: str, runs: int) -> dict:
    from any2jev.model import DecisionModel

    torch.cuda.reset_peak_memory_stats()
    model = DecisionModel.load(checkpoint, dtype=dtype)
    req = {"state": STATE, "questions": QUESTIONS}
    for _ in range(5):
        model.decide(req)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        answers, n_in = model.decide(req)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"model": model.meta.get("base"), "checkpoint": checkpoint, "method": "any2jev, one forward pass",
            "dtype": str(model.compute_dtype), "device": str(model.device), "input_tokens": n_in,
            "latency_ms": round(statistics.median(times), 1), "p95_ms": round(sorted(times)[int(0.95 * len(times)) - 1], 1),
            "answers": answers, "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--checkpoint", default="runs/qwen3-0.6b-public")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", default="runs/vs_recording.json")
    a = ap.parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    left = record_generate(a.base, dtype, "cuda", a.runs, a.max_new_tokens)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"generate: {left['total_ms']} ms, {left['output_tokens']} tokens, first token {left['first_token_ms']} ms, "
          f"format_ok={left['format_ok']}, peak VRAM {left['peak_vram_mb']} MB\n{left['output_text']}\n")
    logits = record_logit_reading(a.base, dtype, "cuda", 20)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"label-logit reading: {logits['total_ms']} ms for 3 questions, peak VRAM {logits['peak_vram_mb']} MB")
    right = record_any2jev(a.checkpoint, a.dtype, 30)
    print(f"any2jev: {right['latency_ms']} ms (p95 {right['p95_ms']}), peak VRAM {right['peak_vram_mb']} MB\n"
          f"{json.dumps(right['answers'])}")
    rec = {"state": STATE, "questions": QUESTIONS, "gpu": torch.cuda.get_device_name(0), "left": left, "logits": logits,
           "right": right}
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
