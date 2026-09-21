"""LoRA + pointer-head training on labelled requests, followed by temperature scaling.

Loss per question = cross-entropy over the option logits, plus optional proper-scoring terms:
``brier_weight`` * multiclass Brier and, for Score questions, ``ordinal_weight`` * ranked probability
score (squared distance between predicted and observed CDF over the ordered levels).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .calibration import fit_temperature, summarize
from .data import Record, load_jsonl, materialize, shuffle_choices
from .model import DecisionModel


@dataclass
class TrainConfig:
    base: str
    data: str
    out: str
    val: str | None = None
    epochs: float = 2.0
    lr: float = 2e-4
    head_lr: float | None = 1e-3
    batch_size: int = 4
    grad_accum: int = 1
    lora_r: int = 16
    lora_alpha: int | None = None
    lora_targets: list[str] | None = None
    head_dim: int = 256
    max_state: int = 1024
    max_branch: int = 1024
    dtype: str = "fp32"
    attn: str | None = None
    device: str | None = None
    seed: int = 0
    brier_weight: float = 0.0
    ordinal_weight: float = 0.0
    shuffle_options: bool = True
    delimiters: str = "auto"
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_every: int = 10
    max_steps: int | None = None
    calibrate: bool = True
    save_every_epoch: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("batch_size", "grad_accum", "log_every", "max_state", "max_branch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.epochs) or self.epochs <= 0:
            raise ValueError("epochs must be finite and positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")


def question_loss(z: torch.Tensor, y: int, qtype: str, brier_w: float, ordinal_w: float) -> torch.Tensor:
    k = z.shape[-1]
    target = torch.tensor([y], device=z.device)
    loss = F.cross_entropy(z[None], target)
    if brier_w > 0 or (qtype == "score" and ordinal_w > 0):
        p = F.softmax(z, -1)
        if brier_w > 0:
            onehot = F.one_hot(target[0], k).to(p.dtype)
            loss = loss + brier_w * ((p - onehot) ** 2).sum()
        if qtype == "score" and ordinal_w > 0 and k > 1:
            cdf = p.cumsum(-1)[:-1]
            observed = (torch.arange(k - 1, device=z.device) >= y).to(p.dtype)
            loss = loss + ordinal_w * ((cdf - observed) ** 2).mean()
    return loss


def _encode_records(model: DecisionModel, records: list[Record], rng: random.Random | None):
    packs, qtypes = [], []
    for rec in records:
        state, specs = materialize(rec)
        if rng is not None:
            specs = shuffle_choices(specs, rng)
        packs.append(model.encode(state, specs))
        qtypes.append([s.qtype for s in specs])
    return packs, qtypes


@torch.no_grad()
def collect_logits(model: DecisionModel, records: list[Record], batch_size: int = 8):
    """Raw logits and labels for every question in ``records`` (no option shuffling)."""
    model.eval()
    logits, labels, qtypes = [], [], []
    for i in range(0, len(records), batch_size):
        packs, qt = _encode_records(model, records[i : i + batch_size], None)
        for rec_logits, pack, types in zip(model.logits(packs), packs, qt):
            for z, y, t in zip(rec_logits, pack.labels, types):
                logits.append(z.float().cpu().numpy())
                labels.append(int(y))
                qtypes.append(t)
    return logits, labels, qtypes


def load_or_create(cfg: TrainConfig) -> DecisionModel:
    from .hub import resolve_model_dir

    base = resolve_model_dir(cfg.base)
    if DecisionModel.is_checkpoint(base):
        return DecisionModel.load(base, device=cfg.device, dtype=cfg.dtype, attn=cfg.attn, trainable=True)
    return DecisionModel.from_base(
        base, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_targets=cfg.lora_targets, head_dim=cfg.head_dim,
        delimiters=cfg.delimiters, dtype=cfg.dtype, attn=cfg.attn, device=cfg.device, max_state=cfg.max_state,
        max_branch=cfg.max_branch)


def train(cfg: TrainConfig, log=print) -> Path:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    records = load_jsonl(cfg.data)
    if not records:
        raise ValueError(f"training data is empty: {cfg.data}")
    val = load_jsonl(cfg.val) if cfg.val else None
    if cfg.val and not val:
        raise ValueError(f"validation data is empty: {cfg.val}")
    model = load_or_create(cfg)
    # A previous calibration no longer describes weights after further training.
    model.temperature = 1.0
    n_params = model.count_parameters()
    log(f"model: {cfg.base} | mode={model.mode} | trainable {n_params['trainable']:,} / {n_params['total']:,} params")
    log(f"data: {len(records)} records" + (f", val {len(val)} records" if val else ""))

    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for p in model.trainable_parameters() if id(p) not in head_ids]
    groups = [{"params": lora_params, "lr": cfg.lr}, {"params": head_params, "lr": cfg.head_lr or cfg.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(records) / (cfg.batch_size * cfg.grad_accum))
    total_steps = cfg.max_steps or max(1, int(round(steps_per_epoch * cfg.epochs)))
    warmup = max(1, int(total_steps * cfg.warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    history = []
    step, micro, t0 = 0, 0, time.time()
    accumulated_questions = 0
    run_loss, run_n, run_correct, run_q = 0.0, 0, 0, 0
    model.train()
    done = False
    epoch = 0
    while not done:
        order = list(range(len(records)))
        rng.shuffle(order)
        for i in range(0, len(order), cfg.batch_size):
            batch = [records[j] for j in order[i : i + cfg.batch_size]]
            packs, qtypes = _encode_records(model, batch, rng if cfg.shuffle_options else None)
            losses = []
            for rec_logits, pack, types in zip(model.logits(packs), packs, qtypes):
                for z, y, t in zip(rec_logits, pack.labels, types):
                    losses.append(question_loss(z, int(y), t, cfg.brier_weight, cfg.ordinal_weight))
                    run_correct += int(z.argmax().item() == y)
                    run_q += 1
            loss_sum = torch.stack(losses).sum()
            loss = loss_sum / len(losses)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            # Sum first, then normalize by the actual question count in this optimizer step.
            # Microbatches can have different record sizes and different numbers of questions.
            loss_sum.backward()
            accumulated_questions += len(losses)
            run_loss += loss.item() * len(losses)
            run_n += len(losses)
            micro += 1
            if micro < cfg.grad_accum and i + cfg.batch_size < len(order):
                continue
            for p in model.trainable_parameters():
                if p.grad is not None:
                    p.grad.div_(accumulated_questions)
            if cfg.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            micro, accumulated_questions = 0, 0
            step += 1
            if step % cfg.log_every == 0 or step == total_steps:
                rec = {"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": run_loss / max(run_n, 1),
                       "train_acc": run_correct / max(run_q, 1), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0, 1)}
                history.append(rec)
                log(f"step {step}/{total_steps} ep {rec['epoch']:.2f} loss {rec['loss']:.4f} acc {rec['train_acc']:.3f} "
                    f"lr {rec['lr']:.2e} {rec['elapsed_s']}s")
                run_loss, run_n, run_correct, run_q = 0.0, 0, 0, 0
            if step >= total_steps:
                done = True
                break
        epoch += 1
        if cfg.save_every_epoch and not done:
            model.save(Path(cfg.out) / f"epoch{epoch}")
        if val and not done:
            _log_eval(model, val, log, f"epoch {epoch}")
            model.train()

    model.eval()
    report: dict = {"config": asdict(cfg), "history": history, "params": n_params, "train_seconds": round(time.time() - t0, 1)}
    if val:
        logits, labels, qtypes = collect_logits(model, val)
        report["val_uncalibrated"] = summarize(logits, labels, 1.0)
        if cfg.calibrate:
            t = fit_temperature(logits, labels)
            model.temperature = t
            report["temperature"] = t
            report["val_calibrated"] = summarize(logits, labels, t)
            log(f"temperature scaling: T={t:.3f} | ECE {report['val_uncalibrated']['ece']:.4f} -> "
                f"{report['val_calibrated']['ece']:.4f} | acc {report['val_calibrated']['accuracy']:.3f}")
    out = model.save(cfg.out)
    (out / "train_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    log(f"saved to {out}")
    return out


def _log_eval(model, val, log, tag):
    logits, labels, _ = collect_logits(model, val)
    s = summarize(logits, labels, 1.0)
    log(f"[{tag}] val acc {s['accuracy']:.3f} nll {s['nll']:.4f} ece {s['ece']:.4f} brier {s['brier']:.4f}")


def calibrate(model_dir: str | Path, val_path: str | Path, device: str | None = None, log=print) -> float:
    """Fit a temperature on ``val_path`` and write it into the checkpoint config."""
    if str(model_dir).startswith("hf://"):
        raise ValueError("calibrate updates a local checkpoint; copy the Hub checkpoint to a local directory first")
    val = load_jsonl(val_path)
    if not val:
        raise ValueError(f"validation data is empty: {val_path}")
    model = DecisionModel.load(model_dir, device=device)
    logits, labels, _ = collect_logits(model, val)
    before = summarize(logits, labels, 1.0)
    t = fit_temperature(logits, labels)
    after = summarize(logits, labels, t)
    log(f"T={t:.3f} | ECE {before['ece']:.4f} -> {after['ece']:.4f} | NLL {before['nll']:.4f} -> {after['nll']:.4f}")
    p = Path(model_dir) / "any2jev.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["temperature"] = t
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return t


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
