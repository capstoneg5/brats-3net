# src/guardrails/clinical_guardrails.py
from __future__ import annotations

import re
from typing import List, Tuple, Optional
from .schemas import GuardrailDecision, RetrievalBundle


# =============================================
# 0) TOPIC GUARDRAIL  (NEW — blocks off-topic)
# =============================================
# Whitelist approach: only allow queries that match
# medical / BraTS / MRI / MedRAG-X topics.

_MEDICAL_PATTERNS = [
    # Brain / neuro anatomy
    r"\bbrain\b", r"\btumor\b", r"\btumour\b", r"\blesion\b",
    r"\bglioma\b", r"\bglioblastoma\b", r"\bmeningioma\b", r"\bneoplasm\b",
    r"\bcancer\b", r"\boncology\b", r"\bneurology\b", r"\bradiology\b",
    r"\bclinical\b", r"\bmedical\b", r"\bpatient\b", r"\banatomical\b",
    r"\banatomy\b", r"\bneuroimaging\b", r"\bcranial\b", r"\bcerebral\b",
    r"\bcortex\b", r"\bcortical\b", r"\bwhite matter\b", r"\bgrey matter\b",
    r"\bgray matter\b", r"\bventricle\b", r"\btemporal\b", r"\bfrontal\b",
    r"\bparietal\b", r"\boccipital\b", r"\bhippocampus\b", r"\bpathology\b",
    r"\bhealth\b",

    # MRI modalities & imaging
    r"\bmri\b", r"\bt1\b", r"\bt2\b", r"\bflair\b", r"\bt1ce\b",
    r"\bt1.weighted\b", r"\bt2.weighted\b", r"\bcontrast\b",
    r"\bnifti\b", r"\bnii\b", r"\bdicom\b", r"\bvolumetric\b",
    r"\bvoxel\b", r"\bslice\b", r"\bscan\b", r"\bimaging\b",
    r"\bimage\b", r"\bmodality\b", r"\bpet scan\b", r"\bct scan\b",

    # BraTS / segmentation
    r"\bbrats\b", r"\bsegment", r"\bwhole.tumor\b", r"\btumor.core\b",
    r"\benhancing\b", r"\bedema\b", r"\bnecrosis\b", r"\bnecrotic\b",
    r"\bwt\b", r"\btc\b", r"\bncr\b",
    r"\bdice\b", r"\biou\b", r"\bu.?net\b", r"\bmonai\b",

    # Clinical metrics & KG terms
    r"\bet[%_]", r"\btc[%_]", r"\bwt_vox\b", r"\bet_pct\b", r"\btc_pct\b",
    r"\bknowledge.graph\b", r"\bkg\b", r"\bneo4j\b", r"\bretrieval\b",
    r"\brag\b", r"\bembedding\b", r"\bsimilar\b", r"\bcompare\b",
    r"\bconfidence\b", r"\bguardrail\b", r"\bevidence\b",
    r"\bscore\b", r"\bsimilarity\b", r"\bpipeline\b",

    # General medical
    r"\bdiagnos", r"\bprogno", r"\bhistolog", r"\bbiopsy\b",
    r"\bmagnetic.resonance\b", r"\broi\b", r"\bbbox\b", r"\bcentroid\b",

    # System self-reference (asking about MedRAG-X itself)
    r"\bmedrag", r"\bwhat can you\b", r"\bhow do you\b",
    r"\bwhat do you\b", r"\bcapabilit", r"\bhelp\b",
    r"\bupload\b", r"\banalyz", r"\bexplain\b",
]

_OFFTOPIC_KEYWORDS = [
    "cricket", "football", "soccer", "basketball", "tennis", "baseball",
    "hockey", "golf", "olympics", "sports", "world cup", "ipl",
    "sachin", "virat", "dhoni", "messi", "ronaldo",
    "movie", "film", "music", "song", "lyrics", "album", "actor",
    "recipe", "cook", "food", "restaurant", "pizza",
    "weather", "climate", "temperature",
    "stock", "bitcoin", "crypto", "trading", "invest",
    "politics", "election", "president", "prime minister", "congress",
    "joke", "poem", "story", "novel", "fiction",
    "game", "play", "video game", "minecraft", "fortnite",
    "travel", "vacation", "hotel", "flight", "tourism",
    "fashion", "celebrity", "gossip", "instagram", "tiktok",
    "homework", "essay", "math problem", "physics problem",
    "code review", "javascript", "python tutorial",
]


def topic_guardrail(query: str) -> GuardrailDecision:
    """
    Ensures the query is related to medical imaging / BraTS / MedRAG-X.
    Blocks off-topic queries (cricket, movies, general knowledge, etc.)
    BEFORE they reach the LLM.
    """
    if not query or not query.strip():
        return GuardrailDecision(
            action="refuse",
            reason="Empty query.",
            safe_reply="Please enter a question related to brain MRI analysis or lesion comparison.",
        )

    q_lower = query.lower().strip()

    # Short greetings / acknowledgments — always allow
    greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "ok", "yes", "no"]
    words = q_lower.split()
    if len(words) <= 3 and any(w in q_lower for w in greetings):
        return GuardrailDecision(action="allow", reason="Greeting/acknowledgment.")

    # Check medical whitelist
    is_medical = any(re.search(p, q_lower) for p in _MEDICAL_PATTERNS)

    # Check off-topic blacklist
    is_offtopic = any(kw in q_lower for kw in _OFFTOPIC_KEYWORDS)

    # Decision logic:
    #   medical + not offtopic  → allow
    #   offtopic + not medical  → block
    #   both or neither         → block if no medical signal

    if is_medical and not is_offtopic:
        return GuardrailDecision(action="allow", reason="On-topic (medical/BraTS).")

    # Build the refusal message
    refusal = (
        "I'm **MedRAG-X**, a clinical reasoning assistant designed exclusively for "
        "**brain tumor MRI analysis** using the BraTS dataset.\n\n"
        "I can help with:\n"
        "- 🧠 **Lesion comparison** — *\"Compare lesion3 to similar lesions\"*\n"
        "- 🖼️ **MRI image analysis** — upload a brain MRI for analysis\n"
        "- 📊 **Tumor metrics** — ET%, TC%, whole tumor volume\n"
        "- 🔬 **Knowledge graph queries** — retrieval, similarity, evidence\n\n"
        "I cannot answer questions about other topics. Please ask something related to brain MRI analysis."
    )

    if is_offtopic:
        return GuardrailDecision(
            action="refuse",
            reason="Off-topic query detected.",
            safe_reply=refusal,
        )

    if not is_medical:
        return GuardrailDecision(
            action="refuse",
            reason="Query does not match any medical/BraTS topic.",
            safe_reply=refusal,
        )

    # Fallback: allow if medical signal present
    return GuardrailDecision(action="allow", reason="Query appears on-topic.")


# =============================================
# 1) Input guardrails  (EXISTING — unchanged)
# =============================================
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
                "I can't help with emergency medical situations. "
                "If this is urgent, contact local emergency services or a licensed clinician immediately."
            ),
        )

    if _match_any(query, _TREATMENT_PATTERNS) or _match_any(query, _DIAGNOSIS_PATTERNS):
        return GuardrailDecision(
            action="refuse",
            reason="User requested diagnosis/treatment/medical advice.",
            safe_reply=(
                "I can't provide diagnosis or treatment advice. "
                "If you have medical concerns, please consult a licensed clinician. "
                "If you want, I can summarize the available lesion/KG facts from your dataset."
            ),
        )

    # allow by default (research/QC questions)
    return GuardrailDecision(action="allow", reason="Query allowed for research summarization.")

# =============================================
# 2) Retrieval guardrails  (EXISTING — unchanged)
# =============================================
def retrieval_guardrail(bundle: RetrievalBundle) -> GuardrailDecision:
    if not bundle.items:
        return GuardrailDecision(
            action="needs_more_info",
            reason="No retrieval results.",
            safe_reply="I couldn't retrieve any evidence from the KG. Try a different lesion id or add more indexed data."
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

    if top > 0.999 and bundle.k >= 5:
        bundle.warnings.append("Very high similarity; embeddings may be overly clustered. Consider checking separation.")

    return GuardrailDecision(action="allow", reason="Retrieval evidence looks sufficient.", metadata={"warnings": bundle.warnings})

# =============================================
# 3) Generation guardrails  (EXISTING — unchanged)
# =============================================
def enforce_grounding_instructions() -> str:
    return (
        "Use only the provided CONTEXT. If something is not explicitly present, say it is not available.\n"
        "Do not provide treatment/diagnosis/dosage.\n"
    )

# =============================================
# 4) Output guardrails  (EXISTING — unchanged)
# =============================================
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
                "I can't provide prescriptive medical advice. "
                "I can only summarize the retrieved evidence and highlight uncertainties."
            ),
        )
    return GuardrailDecision(action="allow", reason="Answer passes safety checks.")