"""any2jev: turn any open model into a Jev-style System One decision model."""

__version__ = "0.1.0"

from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest  # noqa: E402

__all__ = ["__version__", "SystemOneRequest", "NoulQuestion", "ChoiceQuestion", "ScoreQuestion", "DecisionModel"]


def __getattr__(name):  # lazy: importing torch/transformers only when the model is needed
    if name == "DecisionModel":
        from .model import DecisionModel

        return DecisionModel
    raise AttributeError(name)
