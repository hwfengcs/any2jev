"""Pack one state and N questions into a single sequence with a block-causal attention mask.

Layout (``seg`` 0 = state, ``seg`` k = question k; branch positions restart right after the state)::

    <state> s1 .. sS | <q> i1 .. <opt> o1 .. </opt> <opt> .. </opt> <decide> | <q> .. <decide> | ...

Every token attends causally within its own segment and to the whole state; a question never sees
another question. The readout for question k compares the hidden state at its ``<decide>`` token with
the hidden state at each option's ``</opt>`` token (pointer head). Because the state never attends to
the branches, the state's activations are identical with or without them, which is what makes
state-prefix caching exact.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .schema import QuestionSpec

DELIM_NAMES = ("state", "q", "opt", "/opt", "decide")
NEW_DELIM_TOKENS = ["<|a2j:state|>", "<|a2j:q|>", "<|a2j:opt|>", "<|a2j:/opt|>", "<|a2j:decide|>"]
# Rarely used special tokens that already exist in common tokenizers; reusing them needs no new
# embedding rows. Order matches DELIM_NAMES.
REUSABLE_DELIM_SETS: list[list[str]] = [
    ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"],  # Qwen 2.5 / 3 / 3.5
    [f"<|reserved_special_token_{i}|>" for i in (200, 201, 202, 203, 204)],  # Llama 3.x
    [f"<unused{i}>" for i in (90, 91, 92, 93, 94)],  # Gemma 2 / 3
]


class InputTooLongError(ValueError):
    """A request exceeds the model's encoding limits."""


@dataclass
class Delimiters:
    tokens: list[str]
    ids: list[int]
    added: bool

    def __post_init__(self):
        if len(self.tokens) != 5 or len(self.ids) != 5 or any(i is None or i < 0 for i in self.ids):
            raise ValueError(f"need 5 resolvable delimiter tokens, got {self.tokens} -> {self.ids}")

    @property
    def state(self) -> int:
        return self.ids[0]

    @property
    def q(self) -> int:
        return self.ids[1]

    @property
    def opt(self) -> int:
        return self.ids[2]

    @property
    def opt_end(self) -> int:
        return self.ids[3]

    @property
    def decide(self) -> int:
        return self.ids[4]


def resolve_delimiters(tok, strategy: str = "auto") -> Delimiters:
    """Pick the five delimiter tokens for ``tok``.

    ``auto``: reuse a known set of rarely used special tokens if the tokenizer has one, else add new
    tokens. ``reuse``: reuse or fail. ``add``: always add ``NEW_DELIM_TOKENS`` (idempotent).
    """
    if strategy not in ("auto", "reuse", "add"):
        raise ValueError("strategy must be auto | reuse | add")
    vocab = tok.get_vocab()
    if strategy != "add":
        for cand in REUSABLE_DELIM_SETS:
            if all(t in vocab for t in cand):
                return Delimiters(list(cand), [vocab[t] for t in cand], added=False)
        if strategy == "reuse":
            raise ValueError("tokenizer has no known reusable special tokens; use delimiters='add'")
    if all(t in vocab for t in NEW_DELIM_TOKENS):
        return Delimiters(list(NEW_DELIM_TOKENS), [vocab[t] for t in NEW_DELIM_TOKENS], added=True)
    tok.add_special_tokens({"additional_special_tokens": list(NEW_DELIM_TOKENS)}, replace_additional_special_tokens=False)
    ids = tok.convert_tokens_to_ids(NEW_DELIM_TOKENS)
    return Delimiters(list(NEW_DELIM_TOKENS), list(ids), added=True)


def delimiters_from_tokens(tok, tokens: Sequence[str], added: bool) -> Delimiters:
    ids = tok.convert_tokens_to_ids(list(tokens))
    unk = tok.unk_token_id
    if any(i is None or (unk is not None and i == unk) for i in ids):
        raise ValueError(f"delimiter tokens {list(tokens)} are missing from the tokenizer")
    return Delimiters(list(tokens), list(ids), added)


_SPECIAL_RE = re.compile(r"<\|([^|<>]{1,64})\|>")


def _sanitize(text: str) -> str:
    # Fast tokenizers always match added special tokens, even with add_special_tokens=False.
    return _SPECIAL_RE.sub(lambda m: f"<\u00a6{m.group(1)}\u00a6>", text)


def user_tokens(tok, text: str, delims: Delimiters) -> list[int]:
    """Tokenize caller text so it can never contain a delimiter (option boundaries are unforgeable)."""
    ids = tok(_sanitize(text), add_special_tokens=False).input_ids
    bad = set(delims.ids)
    if any(i in bad for i in ids):
        ids = [i for i in ids if i not in bad]
    return ids


@dataclass
class Packed:
    ids: list[int]
    seg: list[int]
    pos: list[int]
    n_state: int
    decide_idx: list[int]
    opt_idx: list[list[int]]
    labels: list[int | None]
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.ids)


def encode(
    tok,
    delims: Delimiters,
    state: str,
    specs: Sequence[QuestionSpec],
    max_state: int = 1024,
    max_branch: int = 1024,
) -> Packed:
    """Pack ``state`` and ``specs`` into one sequence. The state is truncated to ``max_state`` tokens
    (including its delimiter); a branch longer than ``max_branch`` raises ``ValueError``."""
    if not specs:
        raise ValueError("at least one question is required")
    state_tokens = user_tokens(tok, state, delims)
    truncated = len(state_tokens) + 1 > max_state
    s = [delims.state] + state_tokens[: max_state - 1]
    n_state = len(s)
    ids, seg, pos = list(s), [0] * n_state, list(range(n_state))
    decide_idx, opt_idx, labels = [], [], []
    for k, spec in enumerate(specs, start=1):
        if not spec.options:
            raise ValueError(f"question {spec.qid!r} has no options")
        branch = [delims.q] + user_tokens(tok, spec.instructions, delims)
        ends = []
        for opt in spec.options:
            branch += [delims.opt] + user_tokens(tok, opt, delims) + [delims.opt_end]
            ends.append(len(branch) - 1)
        branch.append(delims.decide)
        if len(branch) > max_branch:
            raise InputTooLongError(f"question {spec.qid!r} is {len(branch)} tokens, over the {max_branch}-token branch limit")
        base = len(ids)
        ids += branch
        seg += [k] * len(branch)
        pos += list(range(n_state, n_state + len(branch)))
        decide_idx.append(base + len(branch) - 1)
        opt_idx.append([base + e for e in ends])
        labels.append(spec.label)
    return Packed(ids, seg, pos, n_state, decide_idx, opt_idx, labels, truncated)


def block_mask(segs: Sequence[Sequence[int]], length: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Additive ``[B, 1, L, L]`` mask: query i may attend key j iff ``j <= i`` and ``seg[j] in (0, seg[i])``
    and j is not padding. Padded query rows keep their diagonal so no row is fully masked."""
    b = len(segs)
    s = torch.full((b, length), -1, dtype=torch.long, device=device)
    for i, seg in enumerate(segs):
        s[i, : len(seg)] = torch.as_tensor(seg, dtype=torch.long, device=device)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = (causal[None] & same & valid_key) | torch.eye(length, dtype=torch.bool, device=device)[None]
    return torch.zeros(b, length, length, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


@dataclass
class Row:
    ids: list[int]
    pos: list[int]
    decide: int
    opts: list[int]


def rows_of(p: Packed) -> list[Row]:
    """Split a packed encoding into one causal row per question (state + that branch). Feeding a row
    alone is equivalent to the packed block-causal form for that question on any architecture."""
    rows, start = [], p.n_state
    state_ids, state_pos = p.ids[: p.n_state], p.pos[: p.n_state]
    for d, oi in zip(p.decide_idx, p.opt_idx):
        end = d + 1
        off = p.n_state - start
        rows.append(Row(state_ids + p.ids[start:end], state_pos + p.pos[start:end], d + off, [o + off for o in oi]))
        start = end
    return rows
