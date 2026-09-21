import math

import pydantic
import pytest

from any2jev.schema import (
    SystemOneRequest,
    choice_confidence,
    label_index,
    render,
    to_answers,
    to_specs,
)

DOC_REQUEST = {
    "state": {"document": "I was charged twice. Please fix this ASAP."},
    "model": "jev-latest",
    "questions": {
        "billing": {"type": "noul", "instructions": "Is this ticket about billing?",
                    "criteria": {"true": "Explicitly about charges", "false": "Not about charges"}},
        "dept": {"type": "choice", "instructions": {"question": "Which team?", "focus": "Pick one."},
                 "criteria": {"billing": "Payments, invoicing, refunds", "technical": None, "sales": {"what": "Pricing"}}},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["can wait", "this week", "today"]},
    },
}


def test_specs_map_three_types_onto_options():
    req = SystemOneRequest.model_validate(DOC_REQUEST)
    specs = to_specs(req, {"billing": True, "dept": "technical", "urgency": 2})
    by_id = {s.qid: s for s in specs}
    assert by_id["billing"].options == ["no: Not about charges", "yes: Explicitly about charges"]
    assert by_id["billing"].label == 1
    assert by_id["dept"].options == ["billing: Payments, invoicing, refunds", "technical", "sales: what: Pricing"]
    assert by_id["dept"].instructions == "question: Which team?\nfocus: Pick one."
    assert by_id["dept"].label == 1
    assert by_id["urgency"].keys == ["0", "1", "2"] and by_id["urgency"].label == 2
    assert render(req.state) == "document: I was charged twice. Please fix this ASAP."


def test_answers_match_documented_shapes():
    specs = to_specs(SystemOneRequest.model_validate(DOC_REQUEST))
    ans = to_answers(specs, [[0.05, 0.95], [0.88, 0.12, 0.0], [0.0, 0.95, 0.05]])
    assert ans["billing"] == {"type": "noul", "noul": 0.95}
    assert ans["dept"]["choice"] == "billing"
    assert ans["dept"]["probabilities"] == {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    assert math.isclose(ans["dept"]["confidence"], (0.88 - 1 / 3) / (1 - 1 / 3), abs_tol=1e-3)
    assert math.isclose(ans["urgency"]["score"], 1.05)
    assert ans["urgency"]["legend"] == {"0": "can wait", "1": "this week", "2": "today"}
    assert set(ans["urgency"]) == {"type", "score", "legend", "probabilities", "confidence"}
    assert math.isclose(ans["urgency"]["confidence"], 0.925, abs_tol=1e-3)  # docs show 0.92 for this distribution


def test_choice_confidence_edges():
    assert choice_confidence([1.0]) == 1.0
    assert math.isclose(choice_confidence([0.5, 0.5]), 0.0)
    assert math.isclose(choice_confidence([1.0, 0.0, 0.0]), 1.0)


@pytest.mark.parametrize("bad", [
    {"state": "x", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}}}},
    {"state": "x", "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["only one"]}}},
    {"state": "x", "questions": {}},
    {"state": "x", "questions": {"q": {"type": "bogus", "instructions": "i"}}},
    {"questions": {"q": {"type": "noul", "instructions": "i"}}},
])
def test_invalid_requests_rejected(bad):
    with pytest.raises(pydantic.ValidationError):
        SystemOneRequest.model_validate(bad)


def test_label_index_forms():
    assert label_index("choice", ["a", "b"], "b") == 1
    assert label_index("noul", ["false", "true"], "yes") == 1
    assert label_index("noul", ["false", "true"], False) == 0
    assert label_index("score", ["0", "1", "2"], 2) == 2
    with pytest.raises(ValueError):
        label_index("choice", ["a", "b"], "zzz")
    with pytest.raises(ValueError):
        label_index("score", ["0", "1"], 5)


def test_permuted_spec_keeps_label_key():
    spec = to_specs(SystemOneRequest.model_validate(DOC_REQUEST), {"dept": "technical"})[1]
    perm = [2, 0, 1]
    p = spec.permuted(perm)
    assert p.keys == ["sales", "billing", "technical"] and p.keys[p.label] == "technical"
