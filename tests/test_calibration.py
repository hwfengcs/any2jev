import numpy as np
import pytest

from any2jev.calibration import (
    aurc,
    brier,
    coverage_at_risk,
    ece,
    fit_temperature,
    reliability_bins,
    softmax,
    summarize,
)


def _sample(n=4000, k=4, scale=2.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k)) * scale
    p = softmax(list(z))
    labels = [int(rng.choice(k, p=row)) for row in p]
    return z, labels


def test_fit_temperature_recovers_overconfidence_factor():
    z, labels = _sample()
    t = fit_temperature([row * 3 for row in z], labels)  # logits inflated 3x -> T should be ~3
    assert 2.6 < t < 3.4
    t1 = fit_temperature(list(z), labels)
    assert 0.85 < t1 < 1.15


def test_ece_extremes():
    conf = np.array([0.9] * 100)
    assert ece(conf, np.array([1.0] * 90 + [0.0] * 10)) < 0.02
    assert ece(conf, np.array([0.0] * 100)) > 0.85


@pytest.mark.parametrize("confidence", [0.0, 0.9, 1.0])
def test_adaptive_ece_keeps_constant_confidence_samples(confidence):
    conf = np.full(10, confidence)
    correct = np.zeros(10)
    bins = reliability_bins(conf, correct, adaptive=True)
    assert sum(row["n"] for row in bins) == 10
    assert ece(conf, correct, adaptive=True) == pytest.approx(confidence)


def test_reliability_bins_empty():
    assert reliability_bins(np.array([]), np.array([]), adaptive=True) == []
    assert ece(np.array([]), np.array([]), adaptive=True) == 0.0


def test_brier_bounds():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert brier(probs, [0, 1]) == 0.0
    assert brier(probs, [1, 0]) == 2.0


def test_selective_metrics():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0])
    assert coverage_at_risk(conf, correct, 0.0) == 0.5
    assert 0 < aurc(conf, correct) < 0.5
    assert aurc(conf, np.array([0, 0, 1, 1])) > aurc(conf, correct)


def test_selective_metrics_do_not_split_confidence_ties():
    # A probability threshold cannot accept just the correct member of a tie.
    conf = np.array([0.9, 0.9, 0.5])
    for correct in (np.array([1, 0, 1]), np.array([0, 1, 1])):
        assert coverage_at_risk(conf, correct, 0.05) == 0.0
        assert coverage_at_risk(conf, correct, 0.4) == 1.0
        assert aurc(conf, correct) == pytest.approx((2 * 0.5 + 1 / 3) / 3)


def test_summarize_keys_and_variable_k():
    logits = [np.array([2.0, 0.1]), np.array([0.5, 0.2, 3.0]), np.array([1.0, 1.0, 1.0, 1.0])]
    s = summarize(logits, [0, 2, 1], t=1.0)
    for k in ("n", "accuracy", "nll", "brier", "ece", "ece_adaptive", "aurc", "coverage_at_5pct_risk", "bins"):
        assert k in s
    assert s["n"] == 3 and abs(s["accuracy"] - 2 / 3) < 1e-9
    assert summarize([], [])["n"] == 0
