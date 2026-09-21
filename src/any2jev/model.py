"""Model surgery: causal-LM backbone without its vocabulary head + LoRA + a pointer readout.

``DecisionModel.from_base("Qwen/Qwen3-0.6B")`` loads the base model, drops the LM head, resolves the
delimiter tokens, wraps the backbone in LoRA and attaches a ``PointerHead``. The model never calls
``generate()``: one ``forward()`` over the packed sequence yields every question's probabilities.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from . import __version__
from .packing import (
    Delimiters,
    Packed,
    block_mask,
    delimiters_from_tokens,
    encode,
    resolve_delimiters,
    rows_of,
)
from .schema import QuestionSpec, SystemOneRequest, render, to_answers, to_specs

CONFIG_NAME = "any2jev.json"
CANDIDATE_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",  # llama / qwen / gemma / mistral
    "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",  # falcon / bloom / gpt-neox
    "c_attn", "c_proj", "c_fc",  # gpt2
    "Wqkv", "out_proj", "fc1", "fc2",  # phi / modernbert
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",  # qwen3.5 gated deltanet
]
DTYPES = {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
          "fp16": torch.float16, "float16": torch.float16}


def parse_dtype(d: str | torch.dtype | None) -> torch.dtype:
    if d is None:
        return torch.float32
    if isinstance(d, torch.dtype):
        return d
    return DTYPES[d.lower()]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_backbone(model: nn.Module) -> nn.Module:
    """Return the decoder stack of a ``*ForCausalLM`` model, leaving the vocabulary head behind."""
    get_dec = getattr(model, "get_decoder", None)
    if callable(get_dec):
        try:
            dec = get_dec()
            if isinstance(dec, nn.Module) and dec is not model:
                return dec
        except Exception:  # noqa: BLE001 - fall through to attribute probing
            pass
    for attr in ("model", "transformer", "base_model"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module) and sub is not model:
            return sub
    return model


def text_config(cfg):
    get = getattr(cfg, "get_text_config", None)
    return get() if callable(get) else cfg


def is_hybrid(cfg) -> bool:
    """True for backbones with recurrent / linear-attention layers that cannot honour a custom 4D mask."""
    types = getattr(text_config(cfg), "layer_types", None) or []
    return any(t not in ("full_attention", "sliding_attention") for t in types)


def uses_sliding_attention(cfg) -> bool:
    cfg = text_config(cfg)
    types = getattr(cfg, "layer_types", None)
    if types:
        return "sliding_attention" in types
    return bool(getattr(cfg, "sliding_window", None)) and getattr(cfg, "use_sliding_window", True)


def detect_lora_targets(backbone: nn.Module) -> list[str] | str:
    names = {n.split(".")[-1] for n, m in backbone.named_modules() if isinstance(m, nn.Linear)}
    found = [c for c in CANDIDATE_LORA_TARGETS if c in names]
    return found or "all-linear"


def _module_name(root: nn.Module, target: nn.Module) -> str | None:
    for n, m in root.named_modules():
        if m is target:
            return n
    return None


class PointerHead(nn.Module):
    """Scores each option's ``</opt>`` state against the question's ``<decide>`` state."""

    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.q = nn.Linear(d, dp)
        self.k = nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        nn.init.orthogonal_(self.q.weight)
        with torch.no_grad():
            self.k.weight.copy_(self.q.weight)  # identity-like start: score ~ projected dot product
            self.q.bias.zero_()
            self.k.bias.zero_()

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:  # [d], [K, d] -> [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class DecisionModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tok,
        delims: Delimiters,
        *,
        head: PointerHead | None = None,
        head_dim: int = 256,
        compute_dtype: torch.dtype = torch.float32,
        temperature: float = 1.0,
        max_state: int = 1024,
        max_branch: int = 1024,
        mode: str = "auto",
        meta: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.tok = tok
        self.delims = delims
        self.compute_dtype = compute_dtype
        self.hidden_size = text_config(backbone.config).hidden_size
        self.head = head or PointerHead(self.hidden_size, head_dim)
        self.head_dim = self.head.q.out_features
        self.temperature = float(temperature)
        self.max_state, self.max_branch = max_state, max_branch
        self.hybrid = is_hybrid(backbone.config)
        sliding = uses_sliding_attention(backbone.config)
        # GPT-2 in older supported Transformers versions flattens supplied 4D masks.
        legacy_mask = text_config(backbone.config).model_type == "gpt2"
        if mode not in ("auto", "packed", "rows"):
            raise ValueError("mode must be auto | packed | rows")
        self.mode = ("rows" if self.hybrid or sliding or legacy_mask else "packed") if mode == "auto" else mode
        if self.mode == "packed" and self.hybrid:
            raise ValueError("hybrid (linear-attention) backbones cannot use the packed mask; use mode='rows'")
        if self.mode == "packed" and sliding:
            raise ValueError("sliding-window backbones require their native attention mask; use mode='rows'")
        if self.mode == "packed" and legacy_mask:
            raise ValueError("this backbone does not consistently support a 4D packed mask; use mode='rows'")
        self.meta = meta or {}
        pad = tok.pad_token_id
        if pad is None:
            pad = tok.eos_token_id if tok.eos_token_id is not None else 0
        self.pad_id = int(pad)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_base(
        cls,
        base: str,
        *,
        lora_r: int = 16,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.05,
        lora_targets: list[str] | str | None = None,
        head_dim: int = 256,
        delimiters: str = "auto",
        trainable_delimiters: bool = True,
        dtype: str | torch.dtype | None = "fp32",
        attn: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        max_state: int = 1024,
        max_branch: int = 1024,
        mode: str = "auto",
    ) -> DecisionModel:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = parse_dtype(dtype)
        tok = AutoTokenizer.from_pretrained(base, revision=revision)
        full = AutoModelForCausalLM.from_pretrained(base, revision=revision, torch_dtype=torch_dtype, attn_implementation=attn)
        backbone = extract_backbone(full)
        del full  # the vocabulary head goes with it
        delims = resolve_delimiters(tok, delimiters)
        emb = backbone.get_input_embeddings()
        if len(tok) > emb.num_embeddings:
            backbone.resize_token_embeddings(len(tok))
            emb = backbone.get_input_embeddings()
        extra: dict[str, Any] = {}
        if trainable_delimiters:
            name = _module_name(backbone, emb)
            if name:
                extra["trainable_token_indices"] = {name.split(".")[-1]: list(delims.ids)}
        targets = lora_targets or detect_lora_targets(backbone)
        backbone = _wrap_lora(backbone, lora_r, lora_alpha or 2 * lora_r, lora_dropout, targets, extra)
        meta = {"base": base, "revision": revision, "lora_r": lora_r, "lora_alpha": lora_alpha or 2 * lora_r,
                "lora_targets": targets, "delimiters_strategy": delimiters, "trainable_delimiters": bool(extra)}
        model = cls(backbone, tok, delims, head_dim=head_dim, compute_dtype=torch_dtype, max_state=max_state,
                    max_branch=max_branch, mode=mode, meta=meta)
        return model.to(device or default_device())

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out / "adapter")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()}, out / "head.safetensors")
        self.tok.save_pretrained(out / "tokenizer")
        cfg = {
            "any2jev_version": __version__,
            "base": self.meta.get("base"),
            "revision": self.meta.get("revision"),
            "delimiters": self.delims.tokens,
            "delimiters_added": self.delims.added,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "temperature": self.temperature,
            "max_state": self.max_state,
            "max_branch": self.max_branch,
            "mode": self.mode,
            "hybrid": self.hybrid,
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "meta": self.meta,
        }
        (out / CONFIG_NAME).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        attn: str | None = None,
        trainable: bool = False,
        merge: bool = False,
    ) -> DecisionModel:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(path)
        cfg = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
        torch_dtype = parse_dtype(dtype or cfg.get("compute_dtype", "fp32"))
        tok = AutoTokenizer.from_pretrained(path / "tokenizer")
        full = AutoModelForCausalLM.from_pretrained(cfg["base"], revision=cfg.get("revision"), torch_dtype=torch_dtype,
                                                    attn_implementation=attn)
        backbone = extract_backbone(full)
        del full
        if len(tok) > backbone.get_input_embeddings().num_embeddings:
            backbone.resize_token_embeddings(len(tok))
        backbone = PeftModel.from_pretrained(backbone, path / "adapter", is_trainable=trainable)
        if merge and not trainable:
            backbone = backbone.merge_and_unload()
        delims = delimiters_from_tokens(tok, cfg["delimiters"], cfg["delimiters_added"])
        head = PointerHead(cfg["hidden_size"], cfg["head_dim"])
        head.load_state_dict(load_file(path / "head.safetensors"))
        model = cls(backbone, tok, delims, head=head, compute_dtype=torch_dtype, temperature=cfg.get("temperature", 1.0),
                    max_state=cfg.get("max_state", 1024), max_branch=cfg.get("max_branch", 1024),
                    mode=cfg.get("mode", "auto"), meta=cfg.get("meta", {}))
        model = model.to(device or default_device())
        if not trainable:
            model.eval()
        return model

    @staticmethod
    def is_checkpoint(path: str | Path) -> bool:
        return (Path(path) / CONFIG_NAME).is_file()

    # ------------------------------------------------------------------ encoding
    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    def encode(self, state: str, specs: list[QuestionSpec], max_state: int | None = None,
               max_branch: int | None = None) -> Packed:
        return encode(self.tok, self.delims, state, specs, max_state or self.max_state, max_branch or self.max_branch)

    # ------------------------------------------------------------------ forward
    def logits(self, packs: list[Packed]) -> list[list[torch.Tensor]]:
        """Per record, per question: raw option logits (gradients flow)."""
        if self.mode == "rows":
            return self._logits_rows(packs)
        return self._logits_packed(packs)

    def _logits_packed(self, packs: list[Packed]) -> list[list[torch.Tensor]]:
        dev = self.device
        b, length = len(packs), max(len(p) for p in packs)
        ids = torch.full((b, length), self.pad_id, dtype=torch.long, device=dev)
        pos = torch.zeros((b, length), dtype=torch.long, device=dev)
        for i, p in enumerate(packs):
            ids[i, : len(p)] = torch.as_tensor(p.ids, device=dev)
            pos[i, : len(p)] = torch.as_tensor(p.pos, device=dev)
        mask = block_mask([p.seg for p in packs], length, self.compute_dtype, dev)
        h = self.backbone(input_ids=ids, position_ids=pos, attention_mask=mask, use_cache=False).last_hidden_state.float()
        return [self._readout(h[i], p.decide_idx, p.opt_idx) for i, p in enumerate(packs)]

    def _logits_rows(self, packs: list[Packed]) -> list[list[torch.Tensor]]:
        dev = self.device
        rows, owners = [], []
        for i, p in enumerate(packs):
            for r in rows_of(p):
                rows.append(r)
                owners.append(i)
        length = max(len(r.ids) for r in rows)
        ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long, device=dev)
        pos = torch.zeros((len(rows), length), dtype=torch.long, device=dev)
        att = torch.zeros((len(rows), length), dtype=torch.long, device=dev)
        for i, r in enumerate(rows):
            ids[i, : len(r.ids)] = torch.as_tensor(r.ids, device=dev)
            pos[i, : len(r.pos)] = torch.as_tensor(r.pos, device=dev)
            att[i, : len(r.ids)] = 1
        h = self.backbone(input_ids=ids, position_ids=pos, attention_mask=att, use_cache=False).last_hidden_state.float()
        out: list[list[torch.Tensor]] = [[] for _ in packs]
        for i, (owner, r) in enumerate(zip(owners, rows)):
            out[owner].append(self.head(h[i, r.decide], h[i, torch.as_tensor(r.opts, device=dev)]))
        return out

    def _readout(self, h: torch.Tensor, decide_idx: list[int], opt_idx: list[list[int]]) -> list[torch.Tensor]:
        return [self.head(h[d], h[torch.as_tensor(oi, device=h.device)]) for d, oi in zip(decide_idx, opt_idx)]

    @torch.no_grad()
    def probs(self, packs: list[Packed], temperature: float | None = None) -> list[list[torch.Tensor]]:
        t = self.temperature if temperature is None else temperature
        return [[F.softmax(z / t, -1).cpu() for z in rec] for rec in self.logits(packs)]

    @torch.no_grad()
    def decide(self, req: SystemOneRequest | dict, labels: dict[str, Any] | None = None) -> tuple[dict[str, Any], int]:
        """Answer one wire request. Returns (answers map, input token count)."""
        if isinstance(req, dict):
            req = SystemOneRequest.model_validate(req)
        specs = to_specs(req, labels)
        packed = self.encode(render(req.state), specs)
        probs = self.probs([packed])[0]
        return to_answers(specs, [p.tolist() for p in probs]), len(packed)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return {"total": total, "trainable": trainable}


def _wrap_lora(backbone, r, alpha, dropout, targets, extra):
    from peft import LoraConfig, get_peft_model

    kwargs = dict(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, task_type="FEATURE_EXTRACTION")
    try:
        return get_peft_model(backbone, LoraConfig(**kwargs, **extra))
    except Exception as e:  # noqa: BLE001
        if not extra:
            raise
        warnings.warn(f"trainable delimiter embeddings unavailable ({e!r}); continuing with frozen delimiter rows",
                      stacklevel=2)
        extra.clear()
        return get_peft_model(backbone, LoraConfig(**kwargs))
