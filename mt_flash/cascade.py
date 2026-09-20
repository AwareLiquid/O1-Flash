"""Fast→slow cascade: confidence-threshold routing (the Jev cascade pattern).

The flash model answers everything in one parallel pass; the router keeps
answers whose calibrated confidence clears the threshold and escalates the
rest to a slow handler (a bigger LLM, a human, or deterministic code —
anything the caller provides). This is the same shape as Jev's cascade and
M1's DualSpeedSentry: the slow layer is paid for only when a real
uncertainty earns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .model import O1Flash
from .schema import Question


@dataclass
class RouteResult:
    question_id: str
    handled: bool
    answer: Any
    escalated: bool


SlowHandler = Callable[[str, Question], Any]


class FastSlowRouter:
    def __init__(self, flash: O1Flash, threshold: float,
                 slow: SlowHandler):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.flash = flash
        self.threshold = threshold
        self.slow = slow

    def route(self, state_text: str, questions: list[Question]
              ) -> list[RouteResult]:
        """One parallel pass, then route each answer by its confidence."""
        resp = self.flash.decide(state_text, questions)
        results: list[RouteResult] = []
        for q in questions:
            ans = resp.answers[q.id]
            conf = ans.confidence
            if conf >= self.threshold:
                results.append(RouteResult(q.id, handled=True, answer=ans,
                                           escalated=False))
            else:
                slow_ans = self.slow(state_text, q)
                results.append(RouteResult(q.id, handled=False,
                                           answer=slow_ans, escalated=True))
        return results
