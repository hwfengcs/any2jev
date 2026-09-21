"""Fill the README result blocks from run artefacts, so numbers in the docs always come from files.

    python scripts/fill_readme.py --eval runs/qwen3-0.6b-public/eval.json \
        --baseline runs/qwen3-0.6b-public/baseline_zeroshot.json --latency runs/latency.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- RESULTS:{name} -->\n).*?(\n<!-- /RESULTS:{name} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker RESULTS:{name} not found")
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def fmt(m: dict, keys=("accuracy", "nll", "brier", "ece", "aurc", "coverage_at_5pct_risk")) -> list[str]:
    return [f"{m[k]:.3f}" if k != "coverage_at_5pct_risk" else f"{m[k]:.2f}" for k in keys]


def public_table(ev: dict, base: dict | None, lang: str) -> str:
    hdr = {"en": ("question type", "n", "system", "acc", "NLL", "Brier", "ECE", "AURC", "cov@5%"),
           "zh": ("问题类型", "n", "系统", "acc", "NLL", "Brier", "ECE", "AURC", "cov@5%")}[lang]
    zs = {"en": "zero-shot logits (base)", "zh": "零样本 logits（基座）"}[lang]
    zst = {"en": "zero-shot + temperature", "zh": "零样本 + 温度缩放"}[lang]
    ours = {"en": "**any2jev**", "zh": "**any2jev**"}[lang]
    rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for g in ("overall", "noul", "choice", "score"):
        if g not in ev["metrics"]:
            continue
        m = ev["metrics"][g]
        if base and g in base["metrics"]:
            rows.append(f"| {g} | {m['n']} | {zs} | " + " | ".join(fmt(base["metrics"][g])) + " |")
            rows.append(f"| {g} | {m['n']} | {zst} | " + " | ".join(fmt(base["metrics_T"][g])) + " |")
        rows.append(f"| {g} | {m['n']} | {ours} | " + " | ".join(f"**{x}**" for x in fmt(m)) + " |")
    extra = []
    if "permutation" in ev and ev["permutation"].get("n"):
        p = ev["permutation"]
        extra.append({"en": f"Option-order test on {p['n']} Choice questions: argmax stable in {p['argmax_stable_rate']:.0%} of them, mean max probability spread {p['mean_max_spread']:.3f}.",
                      "zh": f"选项顺序测试（{p['n']} 个 Choice 问题）：argmax 在 {p['argmax_stable_rate']:.0%} 的问题上保持稳定，概率最大波动均值 {p['mean_max_spread']:.3f}。"}[lang])
    if "isolation" in ev:
        extra.append({"en": f"Isolation check: packed vs. separate answers differ by at most {ev['isolation']['max_abs_prob_diff']:.1e}.",
                      "zh": f"隔离检查：打包提问与单独提问的答案最大差异 {ev['isolation']['max_abs_prob_diff']:.1e}。"}[lang])
    extra.append({"en": f"Temperature fitted on validation: T = {ev['temperature']:.2f}. Test set: {ev['n_records']} records, {ev['n_questions']} questions.",
                  "zh": f"验证集拟合的温度 T = {ev['temperature']:.2f}。测试集 {ev['n_records']} 条记录，{ev['n_questions']} 个问题。"}[lang])
    return "\n".join(rows) + "\n\n" + "\n".join(extra)


def latency_table(lat: list[dict], lang: str) -> str:
    hdr = {"en": ("base", "dtype", "device", "input tokens", "questions", "p50 ms", "p95 ms"),
           "zh": ("基座", "精度", "设备", "输入 token", "问题数", "p50 ms", "p95 ms")}[lang]
    rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for r in lat:
        rows.append(f"| {r['base']} | {r['dtype'].replace('torch.', '')} | {r['device']} | {r['input_tokens']} | {r['questions']} | {r['p50_ms']} | {r['p95_ms']} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--latency", default=None, help="JSON list of bench_latency.py outputs")
    a = ap.parse_args()
    ev = json.loads(Path(a.eval).read_text(encoding="utf-8"))
    base = json.loads(Path(a.baseline).read_text(encoding="utf-8")) if a.baseline else None
    lat = json.loads(Path(a.latency).read_text(encoding="utf-8")) if a.latency else None
    for fn, lang in (("README.md", "en"), ("README_zh.md", "zh")):
        p = ROOT / fn
        text = p.read_text(encoding="utf-8")
        text = replace_block(text, "PUBLIC", public_table(ev, base, lang))
        if lat:
            text = replace_block(text, "LATENCY", latency_table(lat, lang))
        p.write_text(text, encoding="utf-8")
        print("updated", fn)


if __name__ == "__main__":
    main()
