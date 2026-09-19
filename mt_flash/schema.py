"""Typed question/answer schema — the O1-Flash API contract (Jev-style).

A request = one piece of state (text/JSON) + a batch of typed questions.
The model returns typed decisions: a probability distribution per Choice,
a level index + distribution per Score, a scalar probability per
Probability question. Every output is schema-bounded by construction:
no free-form text exists anywhere in the output path, so malformed values
are impossible (wrong decisions are still possible — that is what the
calibrated confidence is for).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChoiceQuestion:
    id: str
    options: tuple[str, ...]            # 1..max_options entries

    def __post_init__(self) -> None:
        if not 1 <= len(self.options) <= 255:
            raise ValueError(f"Choice '{self.id}': options must be 1..255")


@dataclass(frozen=True)
class ScoreQuestion:
    id: str
    levels: tuple[str, ...]             # ordered labels, 2..max_score_levels

    def __post_init__(self) -> None:
        if not 2 <= len(self.levels) <= 10:
            raise ValueError(f"Score '{self.id}': levels must be 2..10")


@dataclass(frozen=True)
class ProbabilityQuestion:
    id: str
    prompt: str                         # the yes/no proposition


Question = ChoiceQuestion | ScoreQuestion | ProbabilityQuestion


@dataclass
class ChoiceAnswer:
    id: str
    distribution: list[float]           # len == options, sums to 1
    choice: int                         # argmax index
    confidence: float                   # peakedness of the distribution

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "distribution": self.distribution,
                "choice": self.choice, "confidence": self.confidence}


@dataclass
class ScoreAnswer:
    id: str
    distribution: list[float]
    level: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "distribution": self.distribution,
                "level": self.level, "confidence": self.confidence}


@dataclass
class ProbabilityAnswer:
    id: str
    probability: float
    confidence: float                   # min(p, 1-p) scaled: peakedness

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "probability": self.probability,
                "confidence": self.confidence}


@dataclass
class DecisionResponse:
    answers: dict[str, Any]             # question id -> typed answer
    model: str = "o1-flash"
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        answers = {k: v.to_dict() if hasattr(v, "to_dict") else v
                   for k, v in self.answers.items()}
        return {"model": self.model, "version": self.version,
                "answers": answers}
