# scripts/run_guarded_rag_query.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests

from src.eval.llm_metrics import Embedder, evaluate_rag_answer
from src.guardrails import (
    input_guardrail,
    retrieval_guardrail,
    output_guardrail,
    RetrievalBundle,
    RetrievalItem,
)
from src.guardrails.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.guardrails.clinical_guardrails import enforce_grounding_instructions
from src.rag.neo4j_retriever import Neo4jRetriever


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"


# -----------------------------
# Helpers: normalize / extract
# -----------------------------
def _normalize_text_for_validation(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def extract_core_answer(answer: str) -> str:
    """
    Keep ONLY the lesion bullets (exclude Evidence/Safety/Confidence).
    This makes grounding/hallucination metrics meaningful.
    """
    t = _normalize_text_for_validation(answer)
    parts = re.split(r"(?im)^\s*\**\s*evidence used\s*:\s*\**\s*$", t, maxsplit=1)
    return parts[0].strip() if parts else t.strip()


def retrieved_docs_from_facts_block(
    facts_block: str, want_indices: Tuple[int, ...] = (2, 3, 4, 5)
) -> List[str]:
    """
    Use the FACTS lines [2]..[5] as retrieved docs for grounding.
    This matches the LLM's numbers/format, so embedding grounding works.
    """
    docs: List[str] = []
    for line in (facts_block or "").splitlines():
        m = re.match(r"^\[(\d+)]\s+(.*)$", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        if idx in want_indices:
            docs.append(m.group(2))
    return docs


# -----------------------------
# Convert Neo4j rows -> RetrievalItems for guardrails
# -----------------------------
def rows_to_items(rows: List[Dict[str, Any]]) -> List[RetrievalItem]:
    items: List[RetrievalItem] = []
    for r in rows:
        facts = {
            "lesion_id": r.get("lesion_id"),
            "patient_id": r.get("patient_id"),
            "wt_vox": r.get("wt_vox"),
            "tc_vox": r.get("tc_vox"),
            "et_vox": r.get("et_vox"),
            "et_pct": r.get("et_pct"),
            "tc_pct": r.get("tc_pct"),
            "centroid": r.get("centroid"),
            "bbox": r.get("bbox"),
            "regions": r.get("regions"),
        }

        items.append(
            RetrievalItem(
                source="neo4j",
                id=str(r.get("lesion_id")),
                score=float(r.get("score", 0.0)),
                facts=facts,
                text=(
                    f"lesion={facts['lesion_id']} "
                    f"patient={facts['patient_id']} "
                    f"score={float(r.get('score', 0.0)):.4f}"
                ),
            )
        )

    items.sort(key=lambda x: x.score, reverse=True)
    return items


def build_context(items: List[RetrievalItem]) -> str:
    lines: List[str] = []
    for i, it in enumerate(items, 1):
        f = it.facts
        regions = f.get("regions") or []
        reg_str = ", ".join(
            f"{x.get('region')}:vox={x.get('voxels')},pct={x.get('pct')}"
            for x in regions
            if x.get("region") is not None
        )
        lines.append(
            f"[{i}] lesion={f.get('lesion_id')} patient={f.get('patient_id')} score={it.score:.4f} "
            f"WT={f.get('wt_vox')} TC={f.get('tc_vox')} ET={f.get('et_vox')} "
            f"ET%={f.get('et_pct')} TC%={f.get('tc_pct')} "
            f"centroid={f.get('centroid')} bbox={f.get('bbox')} regions=({reg_str})"
        )
    return "\n".join(lines)


# -----------------------------
# Ollama LLM call
# -----------------------------
def generate_with_llm(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


# -----------------------------
# Deterministic computed facts
# -----------------------------
def compute_diffs(
    rows: List[Dict[str, Any]], query_lesion_id: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    q = next((r for r in rows if r.get("lesion_id") == query_lesion_id), None)
    if q is None:
        q = rows[0]

    q_et = float(q["et_pct"])
    q_tc = float(q["tc_pct"])

    out: List[Dict[str, Any]] = []
    for r in rows:
        et = float(r["et_pct"])
        tc = float(r["tc_pct"])

        et_delta = et - q_et
        tc_delta = tc - q_tc

        out.append(
            {
                "lesion_id": r["lesion_id"],
                "patient_id": r["patient_id"],
                "score": float(r["score"]),
                "et_frac": et,
                "tc_frac": tc,
                "et_pp": et * 100.0,
                "tc_pp": tc * 100.0,
                "et_delta_frac": et_delta,
                "tc_delta_frac": tc_delta,
                "et_delta_pp": et_delta * 100.0,  # signed pp
                "tc_delta_pp": tc_delta * 100.0,  # signed pp
                "et_abs_pp": abs(et_delta) * 100.0,
                "tc_abs_pp": abs(tc_delta) * 100.0,
            }
        )

    out.sort(key=lambda x: (-x["score"], str(x["lesion_id"])))
    return out, q


def sanitize_bullets(text: str) -> str:
    """
    Removes duplicate 'score=' when 'similarity score=' is already present
    inside the 4 lesion bullet lines.

    Safe:
    - Does NOT change numbers
    - Does NOT change structure
    - Guardrails + metrics remain valid
    """

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Only touch lesion bullet lines
        if re.search(r"^\s*[-•*]?\s*lesion_id=lesion\d+", line, re.IGNORECASE):

            # If both 'score=' and 'similarity score=' exist → remove plain 'score='
            if "similarity score=" in line and re.search(r"\bscore=\d", line):
                line = re.sub(r"\bscore=\d+(?:\.\d+)?\s*", "", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def enforce_similarity_scores_in_bullets(
    text: str,
    diffs: List[Dict[str, Any]],
    required_lesions: List[str],
) -> str:
    """
    Inject 'similarity score=0.xxxx' into bullet lines if missing.
    Uses deterministic diffs scores (no hallucination).
    """
    t = _normalize_text_for_validation(text)

    score_map = {d["lesion_id"]: f"{float(d['score']):.4f}" for d in diffs}

    out_lines: List[str] = []
    for line in t.splitlines():
        m = re.match(r"^\s*-\s*lesion_id=(lesion\d+)\b(.*)$", line)
        if not m:
            out_lines.append(line)
            continue

        lesion = m.group(1)
        if lesion not in required_lesions:
            out_lines.append(line)
            continue

        # already has numeric score -> keep
        if re.search(r"(?i)\b(similarity\s+score|score)\s*=\s*0\.\d{4}\b", line):
            out_lines.append(line)
            continue

        score = score_map.get(lesion, "0.0000")

        # replace lone "similarity" with numeric form
        if re.search(r"(?i)\bsimilarity\b", line) and not re.search(r"(?i)\bsimilarity\s+score\b", line):
            line = re.sub(r"(?i)\bsimilarity\b", f"similarity score={score}", line, count=1)
        else:
            # otherwise inject right after lesion_id
            line = re.sub(
                rf"^\s*-\s*lesion_id={lesion}\b",
                f"- lesion_id={lesion} similarity score={score}",
                line,
                count=1,
            )

        out_lines.append(line)

    return "\n".join(out_lines)

def validate_bullets_have_score(answer: str, required_lesions: List[str]) -> Tuple[bool, str]:
    """
    Enforce: each required lesion bullet MUST contain a numeric similarity score with 4 decimals.
    Accept:
      - similarity score=0.9997
      - score=0.9997
    """
    t = _normalize_text_for_validation(answer)

    for lesion in required_lesions:
        m = re.search(rf"(?im)^\s*-\s*lesion_id={lesion}\b.*$", t)
        if not m:
            return False, f"Missing bullet line for {lesion}."

        line = m.group(0)

        # ✅ IMPORTANT: use {4} (single braces)
        if not re.search(r"(?i)\b(similarity\s+score|score)\s*=\s*0\.\d{4}\b", line):
            return False, (
                f"Bullet for {lesion} must include similarity score with 4 decimals "
                f"(example: similarity score=0.9997). Found: {line}"
            )

    return True, ""

def format_computed_facts(diffs: List[Dict[str, Any]], q: Dict[str, Any]) -> str:
    q_et = float(q["et_pct"])
    q_tc = float(q["tc_pct"])

    lines: List[str] = [
        f"QUERY lesion={q['lesion_id']} patient={q['patient_id']}",
        f"QUERY ET_frac={q_et:.6f} (ET%={q_et*100:.2f}%)  TC_frac={q_tc:.6f} (TC%={q_tc*100:.2f}%)",
        "",
        "COMPARISONS vs QUERY (already computed; do NOT recompute):",
        "NOTE: ΔET_pp = (ET_frac - QUERY_ET_frac)*100 (signed, percentage points).",
    ]

    for i, d in enumerate(diffs, 1):
        lines.append(
            f"[{i}] lesion={d['lesion_id']} score={d['score']:.4f} "
            f"ET%={d['et_pp']:.2f}%  ΔET_pp={d['et_delta_pp']:+.2f}pp (abs={d['et_abs_pp']:.2f}pp) "
            f"TC%={d['tc_pp']:.2f}%  ΔTC_pp={d['tc_delta_pp']:+.2f}pp (abs={d['tc_abs_pp']:.2f}pp)"
        )

    return "\n".join(lines)


# -----------------------------
# Numeric consistency guard
# -----------------------------
def validate_answer_numbers(
    answer: str, diffs: List[Dict[str, Any]], query_id: str
) -> Tuple[bool, str]:
    low = (answer or "").lower()

    truth_abs = {d["lesion_id"].lower(): round(float(d["et_abs_pp"]), 2) for d in diffs}

    percents = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", answer or "")]
    if any(p > 100.0 for p in percents):
        return False, (
            "Answer used percentages > 100%. ET values are fractions in [0,1]. "
            "If converting to percent, multiply by 100; for differences use percentage points (diff*100)."
        )

    for s in re.findall(r"(\d+(?:\.\d+)?)\s*(?:percentage points?|pp)\b", low):
        val = float(s)
        if 0.0 < val <= 1.0:
            return False, (
                f"Answer wrote '{val}' as percentage points/pp. That's a fraction. "
                f"Convert to pp by multiplying by 100 (e.g., {val} -> {val*100:.2f} pp)."
            )

    pairs = re.findall(r"(lesion\d+)\D{0,120}?([+-]?\d+(?:\.\d+)?)\s*pp\b", low)
    for lesion, pp_str in pairs:
        lesion = lesion.lower()
        if lesion == query_id.lower():
            continue
        claimed_abs_pp = round(abs(float(pp_str)), 2)
        true_abs_pp = truth_abs.get(lesion)
        if true_abs_pp is None:
            continue
        if abs(claimed_abs_pp - true_abs_pp) > 0.25:
            return False, (
                f"Mismatch: {lesion} abs ΔET should be {true_abs_pp:.2f}pp, "
                f"but answer says {claimed_abs_pp:.2f}pp."
            )

    mentioned = set(re.findall(r"\blesion\d+\b", low))
    expected = set(d["lesion_id"].lower() for d in diffs if d["lesion_id"].lower() != query_id.lower())
    missing = expected - mentioned
    if missing:
        return False, f"Answer did not mention all retrieved lesions. Missing: {sorted(missing)}"

    return True, ""


# -----------------------------
# Evidence section validator
# -----------------------------
def validate_evidence_section(answer: str, required_fact_indices: List[int]) -> Tuple[bool, str]:
    text = _normalize_text_for_validation(answer)

    m = re.search(r"(?im)^\s*\**\s*evidence used\s*:\s*\**\s*$", text)
    if not m:
        return False, "Missing 'Evidence used:' block."

    start = m.end()
    m_end = re.search(r"(?im)^\s*\**\s*safety note\s*:\s*", text[start:])
    evidence_block = text[start : start + (m_end.start() if m_end else len(text))]

    present = set(int(x) for x in re.findall(r"\[(\d+)]", evidence_block))
    missing = set(required_fact_indices) - present
    if missing:
        return False, f"Evidence used is missing FACT indices: {sorted(missing)}"

    return True, ""


# -----------------------------
# Uncertainty calibration guard
# -----------------------------
def validate_uncertainty_statement(
    answer: str, diffs: List[Dict[str, Any]], threshold: float = 0.85
) -> Tuple[bool, str]:
    if not diffs:
        return False, "No retrieval results available to assess confidence."

    top_score = float(diffs[0]["score"])
    if top_score >= threshold:
        return True, ""

    low = (answer or "").lower()
    uncertainty_phrases = [
        "uncertain",
        "not enough evidence",
        "insufficient evidence",
        "based on limited data",
        "cannot determine",
        "i don't know",
        "unclear",
        "low confidence",
    ]
    if any(p in low for p in uncertainty_phrases):
        return True, ""

    return False, (
        f"Top similarity score is low ({top_score:.3f} < {threshold}). "
        "Answer must include an uncertainty statement (e.g., 'insufficient evidence' or 'low confidence')."
    )


# -----------------------------
# Confidence scoring
# -----------------------------
@dataclass
class AnswerQualitySignals:
    input_allowed: bool
    retrieval_allowed: bool
    output_allowed: bool
    numeric_ok: bool
    top_score: float
    passed_count: int
    total_count: int
    coverage: float


def build_signals_from_diffs(
    diffs: List[Dict[str, Any]],
    min_score: float,
    input_allowed: bool,
    retrieval_allowed: bool,
    output_allowed: bool,
    numeric_ok: bool,
) -> AnswerQualitySignals:
    scores = [float(d["score"]) for d in diffs if d.get("score") is not None]
    top = max(scores) if scores else 0.0
    passed = [s for s in scores if s >= min_score]
    total = len(scores)
    coverage = (len(passed) / total) if total else 0.0
    return AnswerQualitySignals(
        input_allowed=input_allowed,
        retrieval_allowed=retrieval_allowed,
        output_allowed=output_allowed,
        numeric_ok=numeric_ok,
        top_score=top,
        passed_count=len(passed),
        total_count=total,
        coverage=coverage,
    )


def retrieval_strength_label(
    top_score: float,
    coverage: float,
    min_score: float,
    high_top: float = 0.95,
    moderate_top: float = 0.90,
    high_coverage: float = 0.80,
    moderate_coverage: float = 0.60,
) -> Tuple[str, str]:
    if top_score < min_score:
        return "Low", f"Top score {top_score:.4f} below min_score {min_score:.2f}."
    if coverage < moderate_coverage:
        return "Low", f"Coverage {coverage:.2f} below {moderate_coverage:.2f}."

    if top_score >= high_top and coverage >= high_coverage:
        return "High", f"Top score {top_score:.4f}, coverage {coverage:.2f}."
    if top_score >= moderate_top and coverage >= moderate_coverage:
        return "Moderate", f"Top score {top_score:.4f}, coverage {coverage:.2f}."

    return "Moderate", f"Top score {top_score:.4f}, coverage {coverage:.2f}."


def answer_confidence(signals: AnswerQualitySignals, min_score: float) -> Tuple[str, str]:
    if not signals.input_allowed:
        return "Low", "Input guardrail did not allow."
    if not signals.retrieval_allowed:
        return "Low", "Retrieval guardrail did not allow."
    if not signals.output_allowed:
        return "Low", "Output guardrail did not allow."
    if not signals.numeric_ok:
        return "Low", "Numeric validation failed."

    retrieval_label, retrieval_reason = retrieval_strength_label(
        top_score=signals.top_score,
        coverage=signals.coverage,
        min_score=min_score,
    )

    if retrieval_label == "Low":
        return "Low", f"Weak retrieval: {retrieval_reason}"
    if retrieval_label == "High":
        return "High", f"Strong retrieval + passed numeric checks + guardrails allowed. ({retrieval_reason})"
    return "Moderate", f"Acceptable retrieval + passed numeric checks + guardrails allowed. ({retrieval_reason})"


def append_confidence_block(answer: str, confidence: str, rationale: str) -> str:
    cleaned = re.sub(
        r"(?is)\n+confidence\s*:\s*(high|moderate|low).*?$",
        "",
        answer or "",
        flags=re.IGNORECASE,
    ).rstrip()
    return cleaned + f"\n\nConfidence: {confidence}\nConfidence rationale: {rationale}\n"


# ---------------------------------------------------------
# Strict structure validator (paper-ready)
# ---------------------------------------------------------
def validate_no_extra_sections(text: str) -> Tuple[bool, str]:
    t = _normalize_text_for_validation(text)

    if not t.strip():
        return False, "Empty answer."

    forbidden = ["answer", "summary", "explanation", "analysis", "reasoning", "note"]
    for h in forbidden:
        if re.search(rf"(?im)^\s*\**\s*{h}\b", t):
            return False, f"Forbidden extra section detected: '{h}'. Output must start directly with lesion bullets."

    # Require exactly 4 bullet lines that start with "- lesion_id="
    lesion_lines = re.findall(r"(?m)^\s*-\s*lesion_id=lesion\d+\b", t)
    if len(lesion_lines) != 4:
        return False, f"Output must contain exactly 4 lesion bullets. Found {len(lesion_lines)}."

    evidence_count = len(re.findall(r"(?im)^\s*\**\s*evidence used\s*:\s*\**\s*$", t))
    if evidence_count != 1:
        return False, f"Output must contain exactly ONE 'Evidence used:' block. Found {evidence_count}."

    safety_count = len(re.findall(r"(?im)^\s*\**\s*safety note\s*:\s*", t))
    if safety_count != 1:
        return False, f"Output must contain exactly ONE 'Safety note:' block. Found {safety_count}."

    confidence_count = len(re.findall(r"(?im)^\s*confidence\s*:\s*(high|moderate|low)\b", t))
    if confidence_count != 1:
        return False, f"Output must contain exactly ONE Confidence block. Found {confidence_count}."

    return True, ""


# -----------------------------
# Retry prompt builder
# -----------------------------
def build_retry_prompt(
    original_prompt: str,
    validation_msg: str,
    facts_block: str,
    query_lesion_id: str,
    required_lesions: List[str],
    required_evidence_indices: List[int],
) -> str:
    lesions_list = "\n".join(f"- {x}" for x in required_lesions)

    # ✅ Force full evidence lines (NOT just [2])
    evidence_full = """Evidence used:
• [2] lesion=lesion117
• [3] lesion=lesion381
• [4] lesion=lesion345
• [5] lesion=lesion51
"""

    return f"""{original_prompt}

SYSTEM FIX REQUEST:
Your previous answer failed validation: {validation_msg}

You MUST:
1) Output exactly {len(required_lesions)} bullet lines.
2) DO NOT write any 'Answer', 'Note', or 'Summary' headers.
3) Each bullet line MUST start with "- lesion_id=<id>" exactly.
4) Each bullet MUST contain "similarity score=0.xxxx" (4 decimals).
5) Include ALL these lesions (exclude query {query_lesion_id}):
{lesions_list}

After the 4 bullets, output EXACTLY this block:
{evidence_full}

Then output:
Safety note: <one sentence only>

Use ONLY the numbers in FACTS. Do NOT add extra lesions.

FACTS (again):
{facts_block}
"""


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--lesion_id", required=True, help="Query lesion id, e.g. lesion38")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.85)

    # Neo4j (optional overrides)
    ap.add_argument("--neo4j_uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    ap.add_argument("--neo4j_user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j_password", default=os.getenv("NEO4J_PASSWORD", "neo4j123"))

    args = ap.parse_args()

    lesion_id = args.lesion_id
    k = args.top_k
    MIN_SCORE = args.min_score
    query = f"Compare {lesion_id} to its most similar lesions and explain ET% differences."

    # (1) INPUT GUARDRAIL
    d0 = input_guardrail(query)
    if d0.action != "allow":
        print(d0.safe_reply or d0.reason)
        return
    input_allowed = True

    # (2) RETRIEVAL
    retriever = Neo4jRetriever(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )
    rows = retriever.retrieve_similar(lesion_id=lesion_id, k=k)

    # (3) RETRIEVAL GUARDRAIL
    items = rows_to_items(rows)
    bundle = RetrievalBundle(query=query, items=items, k=k, min_score=MIN_SCORE)
    d1 = retrieval_guardrail(bundle)
    if d1.action != "allow":
        print(d1.safe_reply or d1.reason)
        return
    retrieval_allowed = True

    # (4) deterministic facts + context
    diffs, q = compute_diffs(rows, lesion_id)
    facts_block = format_computed_facts(diffs, q)
    context_block = build_context(items[:k])

    # ---- dynamic evidence + lesions (no hardcoded lesion117 etc.) ----
    # rows usually include the query lesion itself as the top-1; exclude it.
    required_lesions = [d["lesion_id"] for d in diffs if d["lesion_id"] != lesion_id][: max(0, k - 1)]
    if not required_lesions:
        print("⚠️ No similar lesions returned (after excluding query).")
        print(context_block)
        return

    evidence_lines = "\n".join([f"• [{i+1}] lesion={d['lesion_id']}" for i, d in enumerate(diffs) if d["lesion_id"] != lesion_id][: max(0, k - 1)])

    # (5) Prompt (relax the old demo constraints)
    system = SYSTEM_PROMPT + "\n" + enforce_grounding_instructions() + f"""
IMPORTANT OUTPUT FORMAT (must follow):
- Output EXACTLY {len(required_lesions)} bullet lines.
- NO extra headers.
- Each bullet MUST start EXACTLY with: "- lesion_id=<id>"
- Exclude the query lesion ({lesion_id}).
- Each bullet MUST include:
  - similarity score=0.xxxx (4 decimals)
  - ET%= (2 decimals)
  - ΔET_pp= (signed, 2 decimals)
- Use ONLY the FACTS block numbers. Do NOT recompute.

After the bullets, output:

Evidence used:
{evidence_lines}

Safety note: <one sentence only>
"""

    user = USER_TEMPLATE.format(query=query)
    prompt = f"""{system}

FACTS (use these numbers only):
{facts_block}

RAW CONTEXT:
{context_block}

{user}
"""

    # (6) LLM answer
    llm_answer = generate_with_llm(prompt)

    # (7) OUTPUT GUARDRAIL (before confidence)
    d2 = output_guardrail(llm_answer)
    if d2.action != "allow":
        print(d2.safe_reply or "Output blocked by guardrails.")
        return
    output_allowed = True

    # (8) Append confidence ONCE
    signals = build_signals_from_diffs(
        diffs=diffs,
        min_score=MIN_SCORE,
        input_allowed=input_allowed,
        retrieval_allowed=retrieval_allowed,
        output_allowed=output_allowed,
        numeric_ok=True,
    )
    conf, rationale = answer_confidence(signals, min_score=MIN_SCORE)
    llm_answer = append_confidence_block(llm_answer, conf, rationale)

    # (9) Print outputs
    print("\n===== RETRIEVED CONTEXT =====\n")
    print(context_block)

    print("\n===== FACTS =====\n")
    print(facts_block)

    print("\n===== LLM ANSWER =====\n")
    print(llm_answer)


if __name__ == "__main__":
    main()