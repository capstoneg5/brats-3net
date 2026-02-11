from __future__ import annotations

from src.guardrails.policy import (
    GuardrailDecision, RiskLevel,
    DIAGNOSIS_TRIGGERS, TREATMENT_TRIGGERS, EMERGENCY_TRIGGERS, PRIVACY_TRIGGERS
)

def _contains_any(q: str, terms) -> bool:
    q = q.lower()
    return any(t in q for t in terms)

def precheck_user_query(
    user_query: str,
    has_case_stats: bool,
    has_retrieved_sources: bool,
) -> GuardrailDecision:
    """
    Decides whether the request is safe, and adds prompt rules.
    """

    reasons = []
    q = user_query.lower()

    if _contains_any(q, EMERGENCY_TRIGGERS):
        return GuardrailDecision(
            allowed=False,
            risk=RiskLevel.BLOCK,
            reasons=["Emergency/acute symptom request."],
            safe_response=(
                "I can’t help with emergency medical guidance. If this is urgent, "
                "please contact local emergency services or a licensed clinician immediately."
            ),
        )

    if _contains_any(q, PRIVACY_TRIGGERS):
        return GuardrailDecision(
            allowed=False,
            risk=RiskLevel.BLOCK,
            reasons=["Potential request for sensitive personal/medical identifiers."],
            safe_response="I can’t help with sharing or extracting sensitive personal data.",
        )

    # Diagnosis/treatment requests are high-risk.
    diag = _contains_any(q, DIAGNOSIS_TRIGGERS)
    treat = _contains_any(q, TREATMENT_TRIGGERS)

    # If user asks for diagnosis/treatment, we allow only with strict framing
    # as "research/education" and require evidence grounding.
    if diag or treat:
        reasons.append("Diagnosis/treatment intent detected.")
        rules = (
            "Guardrail rules:\n"
            "- Do NOT claim diagnosis or certainty.\n"
            "- Do NOT prescribe drugs/doses or give personal medical advice.\n"
            "- Provide educational information only.\n"
            "- If case stats are available, report WT/TC/ET volumes/extents only as computed.\n"
            "- If no stats or sources exist, say you cannot determine and ask for required inputs.\n"
            "- Always include a disclaimer to consult a licensed clinician.\n"
        )
        # If we have no evidence, we should be stricter
        if not has_case_stats and not has_retrieved_sources:
            return GuardrailDecision(
                allowed=True,
                risk=RiskLevel.HIGH,
                reasons=reasons + ["No supporting evidence in context."],
                safe_response=None,
                prompt_rules=rules + "- You must refuse to speculate without evidence.\n",
            )
        return GuardrailDecision(
            allowed=True,
            risk=RiskLevel.HIGH,
            reasons=reasons,
            safe_response=None,
            prompt_rules=rules,
        )

    # Normal informational question
    rules = (
        "Guardrail rules:\n"
        "- Stay grounded in provided case stats and retrieved sources.\n"
        "- If information is missing, say what is missing.\n"
        "- Do not hallucinate measurements or cite non-existent sources.\n"
    )

    # If no context at all, warn
    if not has_case_stats and not has_retrieved_sources:
        reasons.append("No case stats or retrieved sources.")
        return GuardrailDecision(
            allowed=True,
            risk=RiskLevel.MEDIUM,
            reasons=reasons,
            safe_response=None,
            prompt_rules=rules + "- Answer generally; do not infer patient-specific facts.\n",
        )

    return GuardrailDecision(
        allowed=True,
        risk=RiskLevel.LOW,
        reasons=["General request."],
        safe_response=None,
        prompt_rules=rules,
    )