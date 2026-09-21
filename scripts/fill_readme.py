"""Fill the README result blocks from run artefacts, so numbers in the docs always come from files.

    python scripts/fill_readme.py --eval runs/qwen3-0.6b-public/eval.json \
        --baseline runs/qwen3-0.6b-public/baseline_zeroshot.json \
        --generate runs/qwen3-0.6b-public/baseline_generate_json.json \
        --vs runs/vs_recording.json --latency runs/latency.json \
        --snake-eval runs/snake/eval.json --snake-play runs/snake/play.log \
        --bases runs/qwen3-0.6b-public runs/qwen3.5-0.8b-synthetic
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


def _load(path: str | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hero_table(vs: dict, gen: dict | None, base: dict | None, ev: dict, lang: str) -> str:
    """The front-page comparison: same base weights, three ways of answering the same questions."""
    z = lang == "zh"
    hdr = (["方案", "怎么出答案", "延迟¹", "格式错误率²", "准确率²", "ECE²", "峰值显存", "每百万请求 GPU 时"] if z else
           ["approach", "how it answers", "latency¹", "format failures²", "accuracy²", "ECE²", "peak VRAM", "GPU-hours / 1M requests"])
    rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    model = vs["left"]["model"]

    def gpu_h(ms):
        return f"~{ms / 3.6:.0f} h"

    rows.append("| " + " | ".join([
        "GPT-4o 级 API，JSON mode" if z else "GPT-4o-class API, JSON mode",
        "云端往返，逐 token 生成" if z else "cloud round trip, token by token",
        "秒级³" if z else "seconds³", "JSON 合法，取值不受约束" if z else "valid JSON, values unchecked", "—", "—",
        "云端" if z else "cloud", "按 token 计费" if z else "per-token billing"]) + " |")
    left = vs["left"]
    if gen:
        g = gen["metrics"]["overall"]
        fail, acc = f"{g['format_failure_rate']:.1%}", f"{g['accuracy']:.3f}"
    else:
        fail, acc = "—", "—"
    rows.append("| " + " | ".join([
        f"{model}，提示它输出 JSON" if z else f"{model}, prompted for JSON",
        "`generate()`，逐 token 解码" if z else "`generate()`, token by token",
        f"{left['total_ms']:.0f} ms", fail, acc, "无概率" if z else "none (no probabilities)",
        f"{left['peak_vram_mb'] / 1024:.1f} GB", gpu_h(left["total_ms"])]) + " |")
    if "logits" in vs and base:
        lg, b = vs["logits"], base["metrics"]["overall"]
        rows.append("| " + " | ".join([
            f"{model}，直接读标签 logits" if z else f"{model}, label-logit reading",
            "每个问题一次 prefill，无需训练" if z else "one prefill per question, no training",
            f"{lg['total_ms']:.0f} ms", "0%", f"{b['accuracy']:.3f}", f"{b['ece']:.3f}",
            f"{lg['peak_vram_mb'] / 1024:.1f} GB", gpu_h(lg["total_ms"])]) + " |")
    r, e = vs["right"], ev["metrics"]["overall"]
    rows.append("| " + " | ".join([
        f"**any2jev（{model}）**" if z else f"**any2jev on {model}**",
        "**单次前向，所有问题一起**" if z else "**one forward pass, all questions at once**",
        f"**{r['latency_ms']:.0f} ms**", "**0%（类型由构造保证）**" if z else "**0% by construction**",
        f"**{e['accuracy']:.3f}**", f"**{e['ece']:.3f}**", f"**{r['peak_vram_mb'] / 1024:.1f} GB**",
        f"**{gpu_h(r['latency_ms'])}**"]) + " |")
    notes = ([
        f"¹ 同一条 3 个问题的客服工单，{vs['gpu']}，fp32，多次运行取中位数（`examples/record_vs.py`）。",
        "² 1,000 个 held-out 问题（boolq、ag_news、banking77、sst5），`any2jev eval` 与 `examples/baseline_*.py` 使用同一份文件。",
        "³ 本仓库未测量。TypeSafe 在发布文章里给出前沿模型端到端 3 到 329 秒；用你自己的 key 跑 `examples/bench_cloud.py` 即可把真实数字填进这一行。",
    ] if z else [
        f"¹ The same 3-question support ticket on one {vs['gpu']}, fp32, median of repeated runs (`examples/record_vs.py`).",
        "² 1,000 held-out questions (boolq, ag_news, banking77, sst5); `any2jev eval` and `examples/baseline_*.py` read the same file.",
        "³ Not measured here. TypeSafe's launch post reports 3 to 329 s end to end for frontier models; run "
        "`examples/bench_cloud.py` with your own key to fill this row with a real number.",
    ])
    return "\n".join(rows) + "\n\n" + "  \n".join(notes)


def public_table(ev: dict, base: dict | None, gen: dict | None, lang: str) -> str:
    z = lang == "zh"
    hdr = (("问题类型", "n", "系统", "acc", "NLL", "Brier", "ECE", "AURC", "cov@5%") if z else
           ("question type", "n", "system", "acc", "NLL", "Brier", "ECE", "AURC", "cov@5%"))
    names = {"gen": "提示输出 JSON（generate）" if z else "prompted for JSON (generate)",
             "zs": "零样本 logits（基座）" if z else "zero-shot logits (base)",
             "zst": "零样本 + 温度缩放" if z else "zero-shot + temperature", "ours": "**any2jev**"}
    rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for g in ("overall", "noul", "choice", "score"):
        if g not in ev["metrics"]:
            continue
        m = ev["metrics"][g]
        if gen and g in gen["metrics"]:
            gm = gen["metrics"][g]
            fail = f" ({gm['format_failure_rate']:.0%} " + ("格式错误" if z else "format failures") + ")"
            rows.append(f"| {g} | {m['n']} | {names['gen']} | {gm['accuracy']:.3f}{fail} | — | — | — | — | — |")
        if base and g in base["metrics"]:
            rows.append(f"| {g} | {m['n']} | {names['zs']} | " + " | ".join(fmt(base["metrics"][g])) + " |")
            rows.append(f"| {g} | {m['n']} | {names['zst']} | " + " | ".join(fmt(base["metrics_T"][g])) + " |")
        rows.append(f"| {g} | {m['n']} | {names['ours']} | " + " | ".join(f"**{x}**" for x in fmt(m)) + " |")
    extra = []
    if "permutation" in ev and ev["permutation"].get("n"):
        p = ev["permutation"]
        extra.append(f"选项顺序测试（{p['n']} 个 Choice 问题）：argmax 在 {p['argmax_stable_rate']:.0%} 的问题上保持稳定，概率最大波动均值 {p['mean_max_spread']:.3f}。" if z else
                      f"Option-order test on {p['n']} Choice questions: argmax stable in {p['argmax_stable_rate']:.0%} of them, mean max probability spread {p['mean_max_spread']:.3f}.")
    if "isolation" in ev:
        extra.append(f"隔离检查：打包提问与单独提问的答案最大差异 {ev['isolation']['max_abs_prob_diff']:.1e}。" if z else
                      f"Isolation check: packed vs. separate answers differ by at most {ev['isolation']['max_abs_prob_diff']:.1e}.")
    extra.append(f"验证集拟合的温度 T = {ev['temperature']:.2f}。测试集 {ev['n_records']} 条记录，{ev['n_questions']} 个问题。" if z else
                  f"Temperature fitted on validation: T = {ev['temperature']:.2f}. Test set: {ev['n_records']} records, {ev['n_questions']} questions.")
    return "\n".join(rows) + "\n\n" + "\n".join(extra)


def latency_table(lat: list[dict], lang: str) -> str:
    hdr = (("基座", "精度", "设备", "输入 token", "问题数", "p50 ms", "p95 ms") if lang == "zh" else
           ("base", "dtype", "device", "input tokens", "questions", "p50 ms", "p95 ms"))
    rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for r in lat:
        rows.append(f"| {r['base']} | {r['dtype'].replace('torch.', '')} | {r['device']} | {r['input_tokens']} | {r['questions']} | {r['p50_ms']} | {r['p95_ms']} |")
    return "\n".join(rows)


def snake_block(sn: dict | None, play: str | None, lang: str) -> str:
    parts = ["![any2jev playing Snake](https://raw.githubusercontent.com/hwfengcs/any2jev/main/docs/snake.gif)", ""]
    if sn and "metrics" in sn:
        m = sn["metrics"]["overall"]
        parts.append(f"held-out 老师走法：准确率 {m['accuracy']:.3f}，ECE {m['ece']:.3f}，{m['n']} 次决策。" if lang == "zh" else
                      f"Held-out teacher moves: accuracy {m['accuracy']:.3f}, ECE {m['ece']:.3f}, {m['n']} decisions.")
    if play:
        parts.append(f"录制对局：{play}" if lang == "zh" else f"Recorded game: {play}")
    return "\n".join(parts)


def bases_table(run_dirs: list[str], lang: str) -> str:
    z = lang == "zh"
    hdr = (("基座", "架构", "模式", "训练数据", "acc", "ECE", "训练耗时", "可训练参数") if z else
           ("base", "architecture", "mode", "trained on", "acc", "ECE", "train time", "trainable params"))
    rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for rd in run_dirs:
        p = Path(rd)
        cfg, rep, ev = _load(p / "any2jev.json"), _load(p / "train_report.json"), _load(p / "eval.json")
        if not (cfg and rep and ev):
            continue
        data = Path(rep["config"]["data"]).parent.name
        arch = ("混合（线性注意力 + 注意力）" if z else "hybrid (linear attention + attention)") if cfg.get("hybrid") else ("纯注意力" if z else "attention-only")
        mode = "rows" if cfg["mode"] == "rows" else "packed"
        m = ev["metrics"]["overall"]
        rows.append(f"| {cfg['base']} | {arch} | {mode} | {data} ({m['n']} q) | {m['accuracy']:.3f} | {m['ece']:.3f} | "
                    f"{rep['train_seconds'] / 60:.0f} min | {rep['params']['trainable'] / 1e6:.1f} M |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--generate", default=None, help="baseline_generate_json.py output")
    ap.add_argument("--vs", default=None, help="record_vs.py output")
    ap.add_argument("--latency", default=None, help="JSON list of bench_latency.py outputs")
    ap.add_argument("--snake-eval", default=None)
    ap.add_argument("--snake-play", default=None, help="play.log from examples/snake.py")
    ap.add_argument("--bases", nargs="*", default=[], help="run directories for the verified-bases table")
    a = ap.parse_args()
    ev = _load(a.eval)
    base, gen, vs, lat, sn = _load(a.baseline), _load(a.generate), _load(a.vs), _load(a.latency), _load(a.snake_eval)
    play = None
    if a.snake_play and Path(a.snake_play).exists():
        lines = [ln for ln in Path(a.snake_play).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.startswith("final score")]
        play = lines[-1] if lines else None
    for fn, lang in (("README.md", "en"), ("README_zh.md", "zh")):
        p = ROOT / fn
        text = p.read_text(encoding="utf-8")
        if vs:
            text = replace_block(text, "HERO", hero_table(vs, gen, base, ev, lang))
        text = replace_block(text, "PUBLIC", public_table(ev, base, gen, lang))
        if lat:
            text = replace_block(text, "LATENCY", latency_table(lat, lang))
        if (ROOT / "docs" / "snake.gif").exists():
            text = replace_block(text, "SNAKE", snake_block(sn, play, lang))
        if a.bases:
            text = replace_block(text, "BASES", bases_table(a.bases, lang))
        p.write_text(text, encoding="utf-8")
        print("updated", fn)


if __name__ == "__main__":
    main()
