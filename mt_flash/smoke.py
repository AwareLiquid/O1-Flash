"""Smoke demo: one state, three typed questions, one parallel pass."""

from mt_flash.model import O1Flash
from mt_flash.schema import (ChoiceQuestion, ProbabilityQuestion,
                             ScoreQuestion)

if __name__ == "__main__":
    m = O1Flash()
    m.eval()
    resp = m.decide(
        "Server down since 03:12; 400 tickets in the infra queue; "
        "on-call not answering.",
        [
            ChoiceQuestion(id="route", options=("billing", "infra", "other")),
            ScoreQuestion(id="sev", levels=("low", "mid", "high")),
            ProbabilityQuestion(id="escalate", prompt="needs escalation now"),
        ],
    )
    import json
    print(json.dumps(resp.to_dict(), indent=2, ensure_ascii=False))
