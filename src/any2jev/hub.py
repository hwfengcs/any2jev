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
    return f"""---
license: apache-2.0
base_model: {base}
tags: [any2jev, system-one, decision-model, jev, calibration, lora]
pipeline_tag: text-classification
---

# {repo_id.split('/')[-1]}

A Jev-style System One decision model made with [any2jev](https://github.com/any2jev/any2jev) from `{base}`.
State in, typed Choice / Score / Noul answers with calibrated probabilities out, in one forward pass.

```bash
pip install any2jev[serve]
any2jev serve hf://{repo_id}
```

Temperature (fitted on validation): {cfg.get('temperature', 1.0):.3f}. LoRA r={cfg.get('meta', {}).get('lora_r')},
pointer head dim {cfg.get('head_dim')}. See the repository for training data and evaluation.
"""
