from __future__ import annotations
from typing import List

from src.guardrails.policy import RiskLevel, GuardrailDecision, DIAGNOSIS_TRIGGERS

BANNED_CERTAINTY = ["definitely", "certainly", "100%", "guarantee", "confirmed"]

def postcheck_llm_output(answer: str) -> GuardrailDecision:
    """
    Simple deterministic post-check.
    Blocks if the model claims diagnosis/treatment certainty.
    """
    a = answer.lower()
    reasons: List[str] = []

    if any(w in a for w in BANNED_CERTAINTY):
        reasons.append("Contains high-certainty medical claim.")

    # If output looks like direct prescription/dose, block
    if "mg" in a and ("take" in a or "dose" in a or "daily" in a):
        reasons.append("Contains medication dosing guidance.")

    # If it uses diagnosis words with certainty
    if any(t in a for t in DIAGNOSIS_TRIGGERS) and any(w in a for w in BANNED_CERTAINTY):
        reasons.append("Diagnosis with certainty.")

    if reasons:
        return GuardrailDecision(
            allowed=False,
            risk=RiskLevel.BLOCK,
            reasons=reasons,
            safe_response=(
                "I can’t provide diagnostic certainty or treatment instructions. "
                "I can summarize the computed imaging metrics and general educational information, "
                "but please consult a licensed clinician for medical decisions."
            ),
        )

    return GuardrailDecision(
        allowed=True,
        risk=RiskLevel.LOW,
        reasons=["Output appears safe."],
    )