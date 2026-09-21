import random

import pytest

from any2jev.data import materialize, shuffle_choices
from any2jev.evaluate import evaluate, isolation_check, permutation_sensitivity


def binary_record():
    return {"state": "The sky is clear.", "questions": {
        "weather": {"type": "choice", "instructions": "Is it sunny?",
                    "criteria": {"yes": None, "no": None}, "label": "yes"}}}


def test_binary_choices_are_shuffled_and_evaluated(tiny_model):
    record = binary_record()
    _, specs = materialize(record)
    rng = random.Random(0)
    orders = set()
    for _ in range(10):
        shuffled = shuffle_choices(specs, rng)[0]
        orders.add(tuple(shuffled.keys))
        assert shuffled.keys[shuffled.label] == "yes"
    assert orders == {("yes", "no"), ("no", "yes")}
    assert permutation_sensitivity(tiny_model, [record])["n"] == 1


def test_isolation_without_eligible_records_is_not_reported_as_zero(tiny_model):
    report = isolation_check(tiny_model, [binary_record()])
    assert report == {"n": 0, "max_abs_prob_diff": None}


def test_evaluation_checks_rows_isolation_and_uses_requested_temperature(tiny_model, monkeypatch):
    record = binary_record()
    record["questions"]["urgent"] = {"type": "noul", "instructions": "Is it urgent?", "label": False}
    monkeypatch.setattr(tiny_model, "mode", "rows")
    original = tiny_model.probs
    temperatures = []

    def probs(packs, temperature=None):
        temperatures.append(temperature)
        return original(packs, temperature=temperature)

    monkeypatch.setattr(tiny_model, "probs", probs)
    report = evaluate(tiny_model, [record], n_perm=2, temperature=2.0)
    assert report["isolation"]["n"] == 1
    assert report["isolation"]["max_abs_prob_diff"] < 1e-5
    assert temperatures and set(temperatures) == {2.0}


def test_empty_evaluation_is_rejected(tiny_model):
    with pytest.raises(ValueError, match="empty"):
        evaluate(tiny_model, [])
