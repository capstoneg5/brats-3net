from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from src.guardrails import (
    input_guardrail,
    retrieval_guardrail,
    output_guardrail,
    RetrievalBundle,
    RetrievalItem,   # if you have it
)


@dataclass
class SafetyResult:
    allowed: bool
    rules_triggered: List[str]
    uncertainty: float
    rationale: str


def estimate_uncertainty(scores: List[float]) -> float:
    if not scores:
        return 1.0
    s = sorted(scores, reverse=True)
    top = s[0]
    gap = (s[0] - s[1]) if len(s) > 1 else top
    u = 1.0 - min(1.0, top) + (0.2 if gap < 0.02 else 0.0)
    return float(min(1.0, max(0.0, u)))


def rule_engine(query: str) -> Tuple[bool, List[str]]:
    t = (query or "").lower()
    triggered: List[str] = []

    if any(k in t for k in ["diagnose", "treatment plan", "prescribe", "medication", "surgery"]):
        triggered.append("No diagnosis/treatment instructions allowed.")
    if any(k in t for k in ["urgent", "emergency", "suicidal"]):
        triggered.append("High-risk user content; advise professional help.")

    return (len(triggered) == 0), triggered


def guard_input_and_retrieval(
    query: str,
    retrieval_items: List[RetrievalItem],
    k: int,
    min_score: float,
) -> SafetyResult:
    """
    Run:
      - your rule engine + uncertainty
      - existing guardrails input + retrieval
    """
    # (A) your rules + uncertainty
    allowed_rules, rules = rule_engine(query)
    scores = [float(it.score) for it in retrieval_items] if retrieval_items else []
    u = estimate_uncertainty(scores)

    # (B) existing INPUT guardrail
    d0 = input_guardrail(query)
    if d0.action != "allow":
        return SafetyResult(
            allowed=False,
            rules_triggered=rules + ["input_guardrail_blocked"],
            uncertainty=u,
            rationale=d0.safe_reply or d0.reason or "Input blocked",
        )

    # (C) existing RETRIEVAL guardrail
    bundle = RetrievalBundle(query=query, items=retrieval_items, k=k, min_score=min_score)
    d1 = retrieval_guardrail(bundle)
    if d1.action != "allow":
        return SafetyResult(
            allowed=False,
            rules_triggered=rules + ["retrieval_guardrail_blocked"],
            uncertainty=u,
            rationale=d1.safe_reply or d1.reason or "Retrieval blocked",
        )

    # combine decision
    allowed = allowed_rules
    rationale = f"allowed={allowed}, uncertainty={u:.2f}, rules={rules}"
    return SafetyResult(allowed=allowed, rules_triggered=rules, uncertainty=u, rationale=rationale)


def guard_output(answer: str) -> SafetyResult:
    """
    Run OUTPUT guardrail after LLM returns text.
    """
    d2 = output_guardrail(answer)
    if d2.action != "allow":
        return SafetyResult(
            allowed=False,
            rules_triggered=["output_guardrail_blocked"],
            uncertainty=0.0,
            rationale=d2.safe_reply or d2.reason or "Output blocked",
        )

    return SafetyResult(
        allowed=True,
        rules_triggered=[],
        uncertainty=0.0,
        rationale="Output allowed",
    )