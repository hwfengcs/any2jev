"""Hugging Face Hub helpers: push a checkpoint (adapter + head + tokenizer + config) and pull it back.

    any2jev push runs/qwen3-0.6b-public --repo <user>/any2jev-qwen3-0.6b
    any2jev serve hf://<user>/any2jev-qwen3-0.6b
"""

from __future__ import annotations

from pathlib import Path

HF_PREFIX = "hf://"


def resolve_model_dir(path: str) -> str:
    """Local directory, or ``hf://repo_id[@revision]`` downloaded to the Hub cache."""
    if not path.startswith(HF_PREFIX):
        return path
    from huggingface_hub import snapshot_download

    repo, _, rev = path[len(HF_PREFIX):].partition("@")
    return snapshot_download(repo, revision=rev or None)


def push(model_dir: str | Path, repo_id: str, private: bool = False, commit_message: str = "any2jev checkpoint") -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
    model_dir = Path(model_dir)
    card = model_dir / "README.md"
    if not card.exists():
        card.write_text(_model_card(model_dir, repo_id), encoding="utf-8")
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id, commit_message=commit_message,
                      ignore_patterns=["*.log", "epoch*/**"])
    return f"https://huggingface.co/{repo_id}"


def _model_card(model_dir: Path, repo_id: str) -> str:
    import json

    cfg = json.loads((model_dir / "any2jev.json").read_text(encoding="utf-8"))
    base = cfg.get("base", "unknown")
    meta = cfg.get("meta", {})
    ev_path, rep_path = model_dir / "eval.json", model_dir / "train_report.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else None
    lines = [
        "---", "license: apache-2.0", f"base_model: {base}", "library_name: any2jev",
        "tags: [any2jev, system-one, decision-model, jev, calibration, lora]", "pipeline_tag: text-classification", "---", "",
        f"# {repo_id.split('/')[-1]}", "",
        f"A Jev-style System One decision model made with [any2jev](https://github.com/hwfengcs/any2jev) from `{base}`.",
        "State in, typed **Choice / Score / Noul** answers with calibrated probabilities out, in one forward pass. No text is generated.",
        "", "```bash", 'pip install "any2jev[serve]"', f"any2jev serve hf://{repo_id}      # POST /v1/systemone, TypeSafe SDK compatible",
        f'any2jev ask hf://{repo_id} --state "My payouts have failed 3 days in a row, fix this ASAP" \\',
        '    --choice "Which team? | billing, technical, sales" --noul "Is this urgent?"', "```", "",
        "## What is in this repo", "",
        f"* `adapter/`: LoRA adapter (r={meta.get('lora_r')}) on `{base}`, trained with the vocabulary head removed",
        f"* `head.safetensors`: the pointer head (dim {cfg.get('head_dim')}) that scores options against the decision token",
        f"* `any2jev.json`: delimiters, mode (`{cfg.get('mode')}`), temperature {cfg.get('temperature', 1.0):.3f} fitted on validation",
        "* `tokenizer/`: the base tokenizer (delimiter tokens reused or added)",
        "* `train_report.json`, `eval.json`: training config, history and held-out metrics",
        "",
    ]
    if rep:
        c = rep.get("config", {})
        data_path = Path(c.get("data", "")).as_posix()
        lines += ["## Training", "",
                  f"* data: `{data_path}`; {rep.get('params', {}).get('trainable', 0) / 1e6:.1f} M trainable parameters, "
                  f"{c.get('epochs')} epoch(s), lr {c.get('lr')}, batch {c.get('batch_size')} x {c.get('grad_accum')}",
                  f"* wall clock: {rep.get('train_seconds', 0) / 60:.0f} min on one consumer GPU", ""]
        if "synthetic" in data_path:
            lines += ["> Trained on any2jev's synthetic support-ticket generator (`any2jev data synthetic`). This checkpoint",
                      "> demonstrates the conversion recipe on this backbone; it is not a general-purpose decision model.", ""]
    if ev:
        lines += ["## Held-out evaluation", "", "| group | n | accuracy | NLL | Brier | ECE | AURC |", "|---|---|---|---|---|---|---|"]
        for g, m in ev["metrics"].items():
            lines.append(f"| {g} | {m['n']} | {m['accuracy']:.3f} | {m['nll']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | {m['aurc']:.3f} |")
        if ev.get("permutation", {}).get("n"):
            p = ev["permutation"]
            lines.append(f"\nOption-order test: argmax stable in {p['argmax_stable_rate']:.0%} of {p['n']} Choice questions.")
        if ev.get("isolation", {}).get("n"):
            lines.append(f"Isolation check (packed vs separate questions): max |dp| = {ev['isolation']['max_abs_prob_diff']:.1e}.")
        lines.append("")
    lines += ["## Limitations", "",
              "Trained on a few thousand labelled decisions from a handful of sources; expect the accuracy above on similar",
              "inputs and lower accuracy off-distribution. Probabilities are calibrated on the validation split, not a guarantee",
              "per answer. Keep arithmetic, dates and counting in code, as Jev's own docs recommend.", "",
              "Independent project, not affiliated with TypeSafe AI. Apache-2.0.", ""]
    return "\n".join(lines)
