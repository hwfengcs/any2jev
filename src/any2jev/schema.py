"""Wire schema for ``POST /v1/systemone`` (TypeSafe / Jev compatible) and the mapping of the three
question types onto the one primitive the model computes: a probability distribution over an
ordered list of option texts.

    Noul   -> options ["no", "yes"]                     answer: noul = p(yes)
    Choice -> options = criteria keys (+ descriptions)  answer: choice, probabilities, confidence
    Score  -> options = ordered level descriptions      answer: score (expected level), legend, probabilities, confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict[str, Any], list[Any], int, float, bool, None]
StateContent = Union[str, dict[str, Any], list[Any]]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 255


class NoulCriteria(BaseModel):
    true: JSONContent = None
    false: JSONContent = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice criteria must have 1..{MAX_CHOICE_OPTIONS} options")
        return self


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: StateContent
    model: str = "any2jev-latest"
    questions: dict[str, Question] = Field(min_length=1)


def render(v: Any, indent: int = 0) -> str:
    """Flatten ``str | object | array`` into the text the model sees. Keys are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (str, int, float)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    if isinstance(v, dict):
        lines = []
        for k, x in v.items():
            if isinstance(x, (dict, list)):
                lines.append(f"{pad}{k}:\n{render(x, indent + 1)}")
            else:
                lines.append(f"{pad}{k}: {render(x)}")
        return "\n".join(lines)
    return str(v)


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


@dataclass
class QuestionSpec:
    """One question as the model sees it: an instruction and an ordered list of option texts."""

    qid: str
    qtype: Literal["noul", "choice", "score"]
    instructions: str
    options: list[str]
    keys: list[str]
    legend: dict[str, str] | None = None
    label: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_options(self) -> int:
        return len(self.options)

    def permuted(self, perm: list[int]) -> QuestionSpec:
        """Re-order options by ``perm`` (new index -> old index); keys and label follow."""
        return replace(
            self,
            options=[self.options[j] for j in perm],
            keys=[self.keys[j] for j in perm],
            label=None if self.label is None else perm.index(self.label),
        )


def label_index(qtype: str, keys: list[str], label: Any) -> int:
    """Map a JSONL label (choice key / bool / level index) to an option index."""
    if label is None:
        raise ValueError("label is required")
    if qtype == "noul":
        if isinstance(label, str):
            value = label.strip().lower()
            if value in ("true", "yes", "1"):
                return 1
            if value in ("false", "no", "0"):
                return 0
        elif isinstance(label, (bool, int, float)) and label in (0, 1):
            return int(label)
        raise ValueError(f"noul label {label!r} must be true/false or 0/1")
    if qtype == "choice":
        if isinstance(label, int) and not isinstance(label, bool) and 0 <= label < len(keys) and str(label) not in keys:
            return label
        if str(label) not in keys:
            raise ValueError(f"label {label!r} is not one of the choice keys {keys}")
        return keys.index(str(label))
    try:
        idx = int(label)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"score label {label!r} must be an integer level index") from e
    if isinstance(label, bool) or (isinstance(label, float) and label != idx):
        raise ValueError(f"score label {label!r} must be an integer level index")
    if not 0 <= idx < len(keys):
        raise ValueError(f"score label {label!r} outside 0..{len(keys) - 1}")
    return idx


def to_specs(req: SystemOneRequest, labels: dict[str, Any] | None = None) -> list[QuestionSpec]:
    specs = []
    for qid, q in req.questions.items():
        label = None if labels is None else labels.get(qid)
        instr = render(q.instructions)
        if q.type == "noul":
            c = q.criteria or NoulCriteria()
            options = [option_text("no", c.false), option_text("yes", c.true)]
            keys = ["false", "true"]
            spec = QuestionSpec(qid, "noul", instr, options, keys)
        elif q.type == "choice":
            keys = list(q.criteria.keys())
            options = [option_text(k, v) for k, v in q.criteria.items()]
            spec = QuestionSpec(qid, "choice", instr, options, keys)
        else:
            options = [render(x) for x in q.criteria]
            keys = [str(i) for i in range(len(options))]
            spec = QuestionSpec(qid, "score", instr, options, keys, legend=dict(zip(keys, options)))
        if label is not None:
            spec.label = label_index(spec.qtype, spec.keys, label)
        specs.append(spec)
    return specs


def choice_confidence(p: list[float]) -> float:
    """Choice statistic used in TypeSafe's public confidence demo; not an accuracy estimate."""
    k = len(p)
    return 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)


def score_confidence(p: list[float]) -> float:
    """Approximate Score confidence from distance to the modal level, normalised by the uniform
    distribution's distance to its center. TypeSafe's exact Score formula is not published."""
    n = len(p)
    if n == 1:
        return 1.0
    mode = max(range(n), key=lambda i: p[i])
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    center = (n - 1) / 2
    uniform_mad = sum(abs(i - center) for i in range(n)) / n
    return max(0.0, 1.0 - dist / uniform_mad)


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def to_answers(specs: list[QuestionSpec], probs: list[list[float]]) -> dict[str, dict[str, Any]]:
    """Build the wire ``answers`` map from per-question probability lists."""
    out: dict[str, dict[str, Any]] = {}
    for spec, p in zip(specs, probs):
        p = [float(x) for x in p]
        if spec.qtype == "noul":
            out[spec.qid] = {"type": "noul", "noul": _r(p[1])}
        elif spec.qtype == "choice":
            best = max(range(len(p)), key=lambda i: p[i])
            out[spec.qid] = {
                "type": "choice",
                "choice": spec.keys[best],
                "probabilities": {k: _r(v) for k, v in zip(spec.keys, p)},
                "confidence": _r(choice_confidence(p)),
            }
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[spec.qid] = {
                "type": "score",
                "score": _r(score),
                "legend": spec.legend,
                "probabilities": {k: _r(v) for k, v in zip(spec.keys, p)},
                "confidence": _r(score_confidence(p)),
            }
    return out


def labels_of(raw_questions: dict[str, Any]) -> dict[str, Any]:
    """Extract ``label`` fields from a labelled-request ``questions`` map (JSONL training format)."""
    return {qid: q["label"] for qid, q in raw_questions.items() if isinstance(q, dict) and "label" in q}
