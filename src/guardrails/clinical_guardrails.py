from __future__ import annotations

import re
from typing import List, Tuple, Optional
from .schemas import GuardrailDecision, RetrievalBundle

# -----------------------------
# 1) Input guardrails
# -----------------------------
_TREATMENT_PATTERNS = [
    r"\bdose\b", r"\bmg\b", r"\btablet\b", r"\bpill\b", r"\bprescribe\b",
    r"\btreatment\b", r"\bchemo\b", r"\bradiation\b", r"\bsurgery\b",
    r"\bmedication\b", r"\bstart taking\b", r"\bwhat should I take\b",
]
_DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b", r"\bis this cancer\b", r"\bstage\b", r"\bprognosis\b",
    r"\bsurvival\b", r"\bdiagnose\b",
]
_EMERGENCY_PATTERNS = [
    r"\bunconscious\b", r"\bseizure\b", r"\bstroke\b", r"\bsevere headache\b",
    r"\bsuicid", r"\bchest pain\b",
]

def _match_any(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def input_guardrail(query: str) -> GuardrailDecision:
    if _match_any(query, _EMERGENCY_PATTERNS):
        return GuardrailDecision(
            action="refuse",
            reason="Potential medical emergency content.",
            safe_reply=(
                "I can’t help with emergency medical situations. "
                "If this is urgent, contact local emergency services or a licensed clinician immediately."
            ),
        )

    if _match_any(query, _TREATMENT_PATTERNS) or _match_any(query, _DIAGNOSIS_PATTERNS):
        return GuardrailDecision(
            action="refuse",
            reason="User requested diagnosis/treatment/medical advice.",
            safe_reply=(
                "I can’t provide diagnosis or treatment advice. "
                "If you have medical concerns, please consult a licensed clinician. "
                "If you want, I can summarize the available lesion/KG facts from your dataset."
            ),
        )

    # allow by default (research/QC questions)
    return GuardrailDecision(action="allow", reason="Query allowed for research summarization.")

# -----------------------------
# 2) Retrieval guardrails
# -----------------------------
def retrieval_guardrail(bundle: RetrievalBundle) -> GuardrailDecision:
    if not bundle.items:
        return GuardrailDecision(
            action="needs_more_info",
            reason="No retrieval results.",
            safe_reply="I couldn’t retrieve any evidence from the KG. Try a different lesion id or add more indexed data."
        )

    top = bundle.items[0].score
    if top < bundle.min_score:
        return GuardrailDecision(
            action="needs_more_info",
            reason=f"Weak retrieval (top score={top:.3f} < min_score={bundle.min_score}).",
            safe_reply=(
                "The retrieved evidence is too weak to answer reliably. "
                "Try increasing k, checking the vector index, or using a different query lesion."
            ),
            metadata={"top_score": top},
        )

    # Optional: flag suspiciously high similarity everywhere (embedding collapse)
    # If your distribution shows mean ~0.97 you may want to warn:
    if top > 0.999 and bundle.k >= 5:
        bundle.warnings.append("Very high similarity; embeddings may be overly clustered. Consider checking separation.")

    return GuardrailDecision(action="allow", reason="Retrieval evidence looks sufficient.", metadata={"warnings": bundle.warnings})

# -----------------------------
# 3) Generation guardrails (prompting)
# -----------------------------
def enforce_grounding_instructions() -> str:
    # You can extend this with “must cite lesion ids” etc.
    return (
        "Use only the provided CONTEXT. If something is not explicitly present, say it is not available.\n"
        "Do not provide treatment/diagnosis/dosage.\n"
    )

# -----------------------------
# 4) Output guardrails
# -----------------------------
_UNSAFE_OUTPUT_PATTERNS = [
    r"\byou should\b",
    r"\bi recommend\b",
    r"\bstart\b.*\bmedication\b",
    r"\btake\b.*\bmg\b",
    r"\bdosage\b",
    r"\bprescrib",
]

def output_guardrail(answer: str) -> GuardrailDecision:
    if _match_any(answer, _UNSAFE_OUTPUT_PATTERNS):
        return GuardrailDecision(
            action="rewrite",
            reason="Answer contains prescriptive medical advice language.",
            safe_reply=(
                "I can’t provide prescriptive medical advice. "
                "I can only summarize the retrieved evidence and highlight uncertainties."
            ),
        )
    return GuardrailDecision(action="allow", reason="Answer passes safety checks.")