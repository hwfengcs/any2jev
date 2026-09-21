"""Calibration metrics and post-hoc temperature scaling. numpy only.

Every question contributes one logit vector of its own length K; helpers take lists of 1-D arrays
(variable K) plus integer labels, and pad with -inf internally.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _pad(logits: Sequence[np.ndarray]) -> np.ndarray:
    k = max(len(z) for z in logits)
    out = np.full((len(logits), k), -np.inf, dtype=np.float64)
    for i, z in enumerate(logits):
        out[i, : len(z)] = np.asarray(z, dtype=np.float64)
    return out


def softmax(logits: Sequence[np.ndarray], t: float = 1.0) -> np.ndarray:
    """Padded ``[N, K_max]`` probabilities (padding columns are exactly 0)."""
    z = _pad(logits) / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logits: Sequence[np.ndarray], labels: Sequence[int], t: float = 1.0) -> float:
    z = _pad(logits) / t
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(labels)), np.asarray(labels)].mean())


def fit_temperature(logits: Sequence[np.ndarray], labels: Sequence[int], lo: float = 0.05, hi: float = 20.0,
                    iters: int = 80) -> float:
    """Scalar temperature minimising NLL (golden-section search over log T; NLL is convex in log T)."""
    if len(logits) == 0:
        return 1.0
    a, b = np.log(lo), np.log(hi)
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(logits, labels, np.exp(c)), nll(logits, labels, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(logits, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(logits, labels, np.exp(d))
    return float(np.exp((a + b) / 2))


def top_label(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(confidence of the argmax, argmax index) for a padded probability matrix."""
    return probs.max(axis=1), probs.argmax(axis=1)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10, adaptive: bool = False) -> list[dict]:
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct, np.float64)
    if adaptive:
        edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                         "conf": float(conf[m].mean()), "acc": float(correct[m].mean())})
    return rows


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10, adaptive: bool = False) -> float:
    n = len(conf)
    return float(sum(r["n"] / n * abs(r["conf"] - r["acc"]) for r in reliability_bins(conf, correct, n_bins, adaptive)))


def brier(probs: np.ndarray, labels: Sequence[int]) -> float:
    """Multiclass Brier score: mean over questions of sum_k (p_k - 1[k = y])^2 (0 = perfect, 2 = worst)."""
    p = np.where(np.isfinite(probs), probs, 0.0)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(labels)), np.asarray(labels)] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve when abstaining on the least confident questions first."""
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - np.asarray(correct, np.float64)[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    """Fraction of questions that can be auto-decided while keeping error rate <= max_risk."""
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - np.asarray(correct, np.float64)[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    ok = np.where(risk <= max_risk)[0]
    return float((ok[-1] + 1) / len(err)) if len(ok) else 0.0


def summarize(logits: Sequence[np.ndarray], labels: Sequence[int], t: float = 1.0, n_bins: int = 10) -> dict:
    """Accuracy, NLL, Brier, ECE (equal-width and adaptive), AURC and reliability bins for one group."""
    if len(logits) == 0:
        return {"n": 0}
    labels = np.asarray(labels)
    probs = softmax(logits, t)
    conf, pred = top_label(probs)
    correct = (pred == labels).astype(np.float64)
    return {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "nll": nll(logits, labels, t),
        "brier": brier(probs, labels),
        "ece": ece(conf, correct, n_bins),
        "ece_adaptive": ece(conf, correct, n_bins, adaptive=True),
        "aurc": aurc(conf, correct),
        "coverage_at_5pct_risk": coverage_at_risk(conf, correct, 0.05),
        "mean_confidence": float(conf.mean()),
        "bins": reliability_bins(conf, correct, n_bins),
    }
