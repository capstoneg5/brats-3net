# scripts/run_rag_eval.py
"""
MedRAG-X — End-to-End RAG Evaluation Pipeline
================================================
Runs the FULL guarded RAG pipeline on N sample lesions and computes:

  RETRIEVAL METRICS:
    • Recall@K, Precision@K, MRR
    • Similarity score distribution (mean, std, min, max)

  RAG GENERATION METRICS:
    • Faithfulness          — does answer ONLY use provided facts?
    • Grounding score       — are all claims backed by evidence indices?
    • Numeric accuracy      — are ET%/ΔET values correct vs deterministic truth?
    • Structure validity    — bullet format, evidence block, safety note
    • Hallucination rate    — fabricated lesion IDs or numbers?
    • Answer completeness   — all retrieved lesions mentioned?

  GUARDRAIL METRICS:
    • Input guardrail pass rate
    • Retrieval guardrail pass rate
    • Output guardrail pass rate
    • Topic guardrail pass rate

  CONFIDENCE CALIBRATION:
    • Brier Score
    • Confidence distribution (High/Moderate/Low)

  LATENCY:
    • Per-step timing (retrieval, LLM, guardrails)
    • End-to-end latency

Usage:
  PYTHONPATH=. python -m scripts.run_rag_eval \
      --n_samples 10 \
      --top_k 5 \
      --min_score 0.70 \
      --out_dir artifacts/rag_eval_results

Outputs:
  artifacts/rag_eval_results/
    rag_eval_summary.json       ← aggregated metrics
    rag_eval_per_query.json     ← per-query details
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


# ═══════════════════════════════════════════════════
#  Imports from existing modules
# ═══════════════════════════════════════════════════

from src.guardrails import (
    input_guardrail,
    retrieval_guardrail,
    output_guardrail,
    RetrievalBundle,
    RetrievalItem,
)
from src.guardrails.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.guardrails.clinical_guardrails import (
    enforce_grounding_instructions,
    topic_guardrail,
)
from src.rag.neo4j_retriever import Neo4jRetriever

# Re-use validators from the guarded RAG script
from scripts.run_guarded_rag_query import (
    compute_diffs,
    format_computed_facts,
    rows_to_items,
    build_context,
    validate_bullets_have_score,
    validate_answer_numbers,
    validate_evidence_section,
    validate_uncertainty_statement,
    validate_no_extra_sections,
    build_signals_from_diffs,
    answer_confidence,
    extract_core_answer,
    retrieved_docs_from_facts_block,
    enforce_similarity_scores_in_bullets,
    sanitize_bullets,
)


# ═══════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


# ═══════════════════════════════════════════════════
#  LLM call
# ═══════════════════════════════════════════════════

def generate_with_llm(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ═══════════════════════════════════════════════════
#  RAG-specific metric functions
# ═══════════════════════════════════════════════════

def compute_faithfulness(answer: str, facts_block: str) -> Dict[str, Any]:
    """
    Faithfulness: does the answer ONLY use numbers from the FACTS block?
    Checks if every numeric claim in the answer appears in the facts.
    """
    core = extract_core_answer(answer)
    if not core.strip():
        return {"score": 0.0, "reason": "Empty answer"}

    # Extract all numbers from answer
    answer_numbers = set(re.findall(r"\d+\.\d{2,4}", core))

    # Extract all numbers from facts
    facts_numbers = set(re.findall(r"\d+\.\d{2,4}", facts_block))

    if not answer_numbers:
        return {"score": 1.0, "reason": "No numeric claims in answer"}

    # How many answer numbers are found in facts?
    grounded = answer_numbers & facts_numbers
    fabricated = answer_numbers - facts_numbers

    score = len(grounded) / len(answer_numbers) if answer_numbers else 1.0

    return {
        "score": round(score, 4),
        "total_numbers": len(answer_numbers),
        "grounded_numbers": len(grounded),
        "fabricated_numbers": len(fabricated),
        "fabricated_values": sorted(fabricated)[:10],  # show up to 10
    }


def compute_hallucination_rate(
    answer: str, diffs: List[Dict[str, Any]], query_id: str
) -> Dict[str, Any]:
    """
    Hallucination detection:
      - Fabricated lesion IDs (not in retrieval set)
      - Fabricated patient IDs
      - Numbers that don't match deterministic truth
    """
    core = extract_core_answer(answer)
    low = core.lower()

    valid_lesions = {d["lesion_id"].lower() for d in diffs}
    valid_patients = {d["patient_id"].lower() for d in diffs if d.get("patient_id")}

    # Find all lesion IDs in answer
    mentioned_lesions = set(re.findall(r"\blesion\d+\b", low))
    fabricated_lesions = mentioned_lesions - valid_lesions

    # Find all patient IDs in answer
    mentioned_patients = set(re.findall(r"\bbrats\w+\b", low))
    fabricated_patients = mentioned_patients - valid_patients if mentioned_patients else set()

    # Check numeric hallucinations (ET/TC values that don't match)
    truth_et = {d["lesion_id"].lower(): round(d["et_pp"], 2) for d in diffs}
    numeric_errors = []

    for lesion, true_et in truth_et.items():
        if lesion == query_id.lower():
            continue
        # Find ET% claimed for this lesion
        m = re.search(rf"{lesion}\D{{0,80}}?(\d+\.\d+)\s*%", low)
        if m:
            claimed = round(float(m.group(1)), 2)
            if abs(claimed - true_et) > 0.5:
                numeric_errors.append({
                    "lesion": lesion,
                    "claimed_et": claimed,
                    "true_et": true_et,
                    "delta": round(abs(claimed - true_et), 2),
                })

    total_claims = len(mentioned_lesions) + len(mentioned_patients) + max(len(truth_et) - 1, 0)
    total_fabricated = len(fabricated_lesions) + len(fabricated_patients) + len(numeric_errors)
    hallucination_rate = total_fabricated / max(total_claims, 1)

    return {
        "hallucination_rate": round(hallucination_rate, 4),
        "fabricated_lesion_ids": sorted(fabricated_lesions),
        "fabricated_patient_ids": sorted(fabricated_patients),
        "numeric_errors": numeric_errors,
        "total_claims": total_claims,
        "total_fabricated": total_fabricated,
    }


def compute_answer_completeness(
    answer: str, diffs: List[Dict[str, Any]], query_id: str
) -> Dict[str, Any]:
    """
    Does the answer mention ALL retrieved lesions (except the query)?
    """
    low = (answer or "").lower()
    expected = {d["lesion_id"].lower() for d in diffs if d["lesion_id"].lower() != query_id.lower()}
    mentioned = set(re.findall(r"\blesion\d+\b", low))
    covered = expected & mentioned
    missing = expected - mentioned

    completeness = len(covered) / len(expected) if expected else 1.0

    return {
        "completeness": round(completeness, 4),
        "expected": sorted(expected),
        "mentioned": sorted(covered),
        "missing": sorted(missing),
    }


def compute_grounding_score(answer: str, required_fact_indices: List[int]) -> Dict[str, Any]:
    """
    Checks the 'Evidence used:' section cites all required FACT indices.
    """
    ok, msg = validate_evidence_section(answer, required_fact_indices)
    # Also check: how many indices are cited vs expected
    m = re.search(r"(?im)evidence used\s*:", answer or "")
    if not m:
        return {"score": 0.0, "cited_indices": [], "expected_indices": required_fact_indices, "reason": "No evidence block"}

    evidence_block = (answer or "")[m.end():]
    cited = sorted(set(int(x) for x in re.findall(r"\[(\d+)]", evidence_block)))
    expected = set(required_fact_indices)
    overlap = set(cited) & expected

    score = len(overlap) / len(expected) if expected else 1.0
    return {
        "score": round(score, 4),
        "cited_indices": cited,
        "expected_indices": required_fact_indices,
        "missing_indices": sorted(expected - set(cited)),
        "valid": ok,
    }


# ═══════════════════════════════════════════════════
#  Single-query RAG evaluation
# ═══════════════════════════════════════════════════

def evaluate_single_query(
    lesion_id: str,
    retriever: Neo4jRetriever,
    top_k: int = 5,
    min_score: float = 0.70,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Run the full guarded RAG pipeline for one lesion and compute all metrics.
    """
    result: Dict[str, Any] = {
        "lesion_id": lesion_id,
        "status": "success",
        "timing": {},
        "guardrails": {},
        "retrieval_metrics": {},
        "rag_metrics": {},
        "confidence": {},
    }

    query = f"Compare {lesion_id} to its most similar lesions and explain ET% differences."

    # ── (1) TOPIC GUARDRAIL ──
    t_topic = topic_guardrail(query)
    result["guardrails"]["topic"] = t_topic.action

    # ── (2) INPUT GUARDRAIL ──
    t0 = time.time()
    d0 = input_guardrail(query)
    result["timing"]["input_guardrail"] = time.time() - t0
    result["guardrails"]["input"] = d0.action

    if d0.action != "allow":
        result["status"] = "blocked_input"
        return result

    # ── (3) RETRIEVAL ──
    t1 = time.time()
    try:
        rows = retriever.retrieve_similar(lesion_id=lesion_id, k=top_k)
    except Exception as e:
        result["status"] = f"retrieval_error: {e}"
        return result
    result["timing"]["retrieval"] = time.time() - t1

    if not rows:
        result["status"] = "no_results"
        return result

    # ── Retrieval metrics ──
    scores = [float(r["score"]) for r in rows]
    non_self = [r for r in rows if r.get("lesion_id") != lesion_id]
    non_self_scores = [float(r["score"]) for r in non_self]

    result["retrieval_metrics"] = {
        "n_retrieved": len(rows),
        "n_non_self": len(non_self),
        "top_score": max(scores),
        "min_score": min(scores),
        "mean_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "scores_above_threshold": sum(1 for s in scores if s >= min_score),
        "all_scores": [round(s, 4) for s in scores],
    }

    # Same-patient retrieval (recall-like)
    query_patient = next((r["patient_id"] for r in rows if r.get("lesion_id") == lesion_id), None)
    if query_patient:
        same_patient = [r for r in non_self if r.get("patient_id") == query_patient]
        result["retrieval_metrics"]["same_patient_retrieved"] = len(same_patient)

    # ── (4) RETRIEVAL GUARDRAIL ──
    items = rows_to_items(rows)
    bundle = RetrievalBundle(query=query, items=items, k=top_k, min_score=min_score)
    d1 = retrieval_guardrail(bundle)
    result["guardrails"]["retrieval"] = d1.action

    if d1.action != "allow":
        result["status"] = "blocked_retrieval"
        return result

    # ── (5) Compute deterministic facts ──
    diffs, q = compute_diffs(rows, lesion_id)
    facts_block = format_computed_facts(diffs, q)
    context_block = build_context(items[:top_k])

    required_lesions = [d["lesion_id"] for d in diffs if d["lesion_id"] != lesion_id][:max(0, top_k - 1)]
    if not required_lesions:
        result["status"] = "no_similar_lesions"
        return result

    # Fact indices (e.g., [2], [3], [4], [5] for top_k=5 excluding self)
    required_fact_indices = list(range(2, 2 + len(required_lesions)))

    evidence_lines = "\n".join([
        f"• [{i+1}] lesion={d['lesion_id']}"
        for i, d in enumerate(diffs) if d["lesion_id"] != lesion_id
    ][:max(0, top_k - 1)])

    # ── (6) Build prompt ──
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

    # ── (7) LLM generation ──
    t2 = time.time()
    try:
        llm_answer = generate_with_llm(prompt)
    except Exception as e:
        result["status"] = f"llm_error: {e}"
        result["timing"]["llm"] = time.time() - t2
        return result
    result["timing"]["llm"] = time.time() - t2

    # ── (8) Post-process ──
    llm_answer = sanitize_bullets(llm_answer)
    llm_answer = enforce_similarity_scores_in_bullets(llm_answer, diffs, required_lesions)

    # ── (9) OUTPUT GUARDRAIL ──
    d2 = output_guardrail(llm_answer)
    result["guardrails"]["output"] = d2.action

    if d2.action != "allow":
        result["status"] = "blocked_output"
        result["rag_metrics"]["output_blocked_reason"] = d2.reason
        return result

    # ══════════════════════════════════════════
    #  RAG QUALITY METRICS
    # ══════════════════════════════════════════

    rag = {}

    # (A) Structure validity
    struct_ok, struct_msg = validate_no_extra_sections(llm_answer)
    rag["structure_valid"] = struct_ok
    rag["structure_msg"] = struct_msg if not struct_ok else ""

    # (B) Bullet score format
    bullet_ok, bullet_msg = validate_bullets_have_score(llm_answer, required_lesions)
    rag["bullet_scores_valid"] = bullet_ok
    rag["bullet_msg"] = bullet_msg if not bullet_ok else ""

    # (C) Numeric accuracy
    num_ok, num_msg = validate_answer_numbers(llm_answer, diffs, lesion_id)
    rag["numeric_accurate"] = num_ok
    rag["numeric_msg"] = num_msg if not num_ok else ""

    # (D) Evidence grounding
    rag["grounding"] = compute_grounding_score(llm_answer, required_fact_indices)

    # (E) Uncertainty statement
    uncert_ok, uncert_msg = validate_uncertainty_statement(llm_answer, diffs, threshold=0.85)
    rag["uncertainty_valid"] = uncert_ok
    rag["uncertainty_msg"] = uncert_msg if not uncert_ok else ""

    # (F) Faithfulness
    rag["faithfulness"] = compute_faithfulness(llm_answer, facts_block)

    # (G) Hallucination detection
    rag["hallucination"] = compute_hallucination_rate(llm_answer, diffs, lesion_id)

    # (H) Answer completeness
    rag["completeness"] = compute_answer_completeness(llm_answer, diffs, lesion_id)

    # (I) Optional: semantic faithfulness via llm_metrics
    try:
        from src.eval.llm_metrics import Embedder, evaluate_rag_answer
        core = extract_core_answer(llm_answer)
        docs = retrieved_docs_from_facts_block(facts_block, tuple(required_fact_indices))
        if docs and core.strip():
            emb = Embedder()
            llm_eval = evaluate_rag_answer(
                query=query,
                answer=core,
                retrieved_docs=docs,
                embedder=emb,
            )
            rag["llm_metrics"] = llm_eval
    except ImportError:
        rag["llm_metrics"] = {"note": "src.eval.llm_metrics not available"}
    except Exception as e:
        rag["llm_metrics"] = {"error": str(e)}

    result["rag_metrics"] = rag

    # ── (10) Confidence ──
    signals = build_signals_from_diffs(
        diffs=diffs,
        min_score=min_score,
        input_allowed=True,
        retrieval_allowed=True,
        output_allowed=(d2.action == "allow"),
        numeric_ok=num_ok,
    )
    conf, rationale = answer_confidence(signals, min_score=min_score)
    result["confidence"] = {
        "label": conf,
        "rationale": rationale,
        "top_score": signals.top_score,
        "coverage": signals.coverage,
        "passed_count": signals.passed_count,
        "total_count": signals.total_count,
    }

    # ── Total timing ──
    result["timing"]["total"] = sum(result["timing"].values())

    return result


# ═══════════════════════════════════════════════════
#  Aggregate metrics across all queries
# ═══════════════════════════════════════════════════

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from per-query results."""
    successful = [r for r in results if r["status"] == "success"]
    n_total = len(results)
    n_success = len(successful)

    if not successful:
        return {
            "n_total": n_total,
            "n_successful": 0,
            "error": "No successful queries",
            "failure_reasons": [r["status"] for r in results],
        }

    # ── Guardrail pass rates ──
    guardrail_rates = {}
    for gtype in ["topic", "input", "retrieval", "output"]:
        vals = [r["guardrails"].get(gtype, "n/a") for r in results if gtype in r.get("guardrails", {})]
        if vals:
            guardrail_rates[f"{gtype}_pass_rate"] = round(sum(1 for v in vals if v == "allow") / len(vals), 4)

    # ── Retrieval metrics ──
    all_top_scores = [r["retrieval_metrics"]["top_score"] for r in successful if r.get("retrieval_metrics")]
    all_mean_scores = [r["retrieval_metrics"]["mean_score"] for r in successful if r.get("retrieval_metrics")]

    retrieval_summary = {
        "mean_top_score": round(float(np.mean(all_top_scores)), 4) if all_top_scores else 0.0,
        "std_top_score": round(float(np.std(all_top_scores)), 4) if all_top_scores else 0.0,
        "mean_mean_score": round(float(np.mean(all_mean_scores)), 4) if all_mean_scores else 0.0,
        "min_top_score": round(float(np.min(all_top_scores)), 4) if all_top_scores else 0.0,
        "max_top_score": round(float(np.max(all_top_scores)), 4) if all_top_scores else 0.0,
    }

    # ── RAG quality metrics ──
    rag_metrics = [r.get("rag_metrics", {}) for r in successful if r.get("rag_metrics")]

    # Binary pass rates
    structure_pass = sum(1 for m in rag_metrics if m.get("structure_valid")) / max(len(rag_metrics), 1)
    bullet_pass = sum(1 for m in rag_metrics if m.get("bullet_scores_valid")) / max(len(rag_metrics), 1)
    numeric_pass = sum(1 for m in rag_metrics if m.get("numeric_accurate")) / max(len(rag_metrics), 1)
    uncertainty_pass = sum(1 for m in rag_metrics if m.get("uncertainty_valid")) / max(len(rag_metrics), 1)

    # Continuous scores
    faithfulness_scores = [m["faithfulness"]["score"] for m in rag_metrics if "faithfulness" in m and "score" in m.get("faithfulness", {})]
    grounding_scores = [m["grounding"]["score"] for m in rag_metrics if "grounding" in m and "score" in m.get("grounding", {})]
    hallucination_rates = [m["hallucination"]["hallucination_rate"] for m in rag_metrics if "hallucination" in m]
    completeness_scores = [m["completeness"]["completeness"] for m in rag_metrics if "completeness" in m]

    rag_summary = {
        "structure_pass_rate": round(structure_pass, 4),
        "bullet_format_pass_rate": round(bullet_pass, 4),
        "numeric_accuracy_rate": round(numeric_pass, 4),
        "uncertainty_valid_rate": round(uncertainty_pass, 4),
        "faithfulness_mean": round(float(np.mean(faithfulness_scores)), 4) if faithfulness_scores else 0.0,
        "faithfulness_std": round(float(np.std(faithfulness_scores)), 4) if faithfulness_scores else 0.0,
        "grounding_mean": round(float(np.mean(grounding_scores)), 4) if grounding_scores else 0.0,
        "grounding_std": round(float(np.std(grounding_scores)), 4) if grounding_scores else 0.0,
        "hallucination_rate_mean": round(float(np.mean(hallucination_rates)), 4) if hallucination_rates else 0.0,
        "completeness_mean": round(float(np.mean(completeness_scores)), 4) if completeness_scores else 0.0,
    }

    # LLM metrics (if available)
    llm_metrics_available = [m.get("llm_metrics", {}) for m in rag_metrics if isinstance(m.get("llm_metrics"), dict) and "error" not in m.get("llm_metrics", {})]
    if llm_metrics_available and "answer_relevance" in llm_metrics_available[0]:
        rag_summary["llm_answer_relevance_mean"] = round(float(np.mean([m.get("answer_relevance", 0) for m in llm_metrics_available])), 4)
        rag_summary["llm_faithfulness_mean"] = round(float(np.mean([m.get("faithfulness", 0) for m in llm_metrics_available])), 4)

    # ── Confidence calibration ──
    CONF_MAP = {"High": 0.90, "Moderate": 0.65, "Low": 0.30}
    predicted_probs = []
    outcomes = []
    conf_counts = {"High": 0, "Moderate": 0, "Low": 0}

    for r in successful:
        conf = r.get("confidence", {})
        label = conf.get("label", "Low")
        conf_counts[label] = conf_counts.get(label, 0) + 1
        predicted_probs.append(CONF_MAP.get(label, 0.30))

        # Outcome: success if numeric + structure + grounding all pass
        rm = r.get("rag_metrics", {})
        outcome = int(
            rm.get("numeric_accurate", False) and
            rm.get("structure_valid", False) and
            rm.get("grounding", {}).get("valid", False)
        )
        outcomes.append(outcome)

    brier = float(np.mean((np.array(predicted_probs) - np.array(outcomes)) ** 2)) if predicted_probs else float("nan")

    confidence_summary = {
        "brier_score": round(brier, 4),
        "brier_interpretation": (
            "excellent" if brier < 0.05 else
            "good" if brier < 0.10 else
            "fair" if brier < 0.20 else "poor"
        ),
        "confidence_distribution": conf_counts,
        "outcome_rate": round(float(np.mean(outcomes)), 4) if outcomes else 0.0,
    }

    # ── Latency ──
    all_timings = [r.get("timing", {}) for r in successful]
    timing_summary = {}
    for step in ["input_guardrail", "retrieval", "llm", "total"]:
        vals = [t.get(step, 0) for t in all_timings if step in t]
        if vals:
            timing_summary[f"{step}_mean_s"] = round(float(np.mean(vals)), 3)
            timing_summary[f"{step}_std_s"] = round(float(np.std(vals)), 3)
            timing_summary[f"{step}_max_s"] = round(float(np.max(vals)), 3)

    return {
        "n_total": n_total,
        "n_successful": n_success,
        "success_rate": round(n_success / n_total, 4) if n_total else 0.0,
        "guardrail_rates": guardrail_rates,
        "retrieval": retrieval_summary,
        "rag_quality": rag_summary,
        "confidence_calibration": confidence_summary,
        "latency": timing_summary,
    }


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="MedRAG-X End-to-End RAG Evaluation")
    ap.add_argument("--n_samples", type=int, default=10, help="Number of lesions to evaluate")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.70)
    ap.add_argument("--lesion_ids", type=str, default=None,
                    help="Comma-separated lesion IDs to evaluate (optional)")
    ap.add_argument("--neo4j_uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    ap.add_argument("--neo4j_user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j_password", default=os.getenv("NEO4J_PASSWORD"))
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts/rag_eval_results"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.neo4j_password:
        raise RuntimeError("Missing Neo4j password. Set NEO4J_PASSWORD or pass --neo4j_password.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MedRAG-X — End-to-End RAG Evaluation")
    print("=" * 60)
    print(f"  Neo4j:      {args.neo4j_uri}")
    print(f"  LLM:        {OLLAMA_MODEL}")
    print(f"  top_k:      {args.top_k}")
    print(f"  min_score:  {args.min_score}")
    print(f"  n_samples:  {args.n_samples}")
    print()

    # ── Connect to Neo4j ──
    retriever = Neo4jRetriever(
        uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password,
    )

    # ── Get sample lesion IDs ──
    if args.lesion_ids:
        sample_ids = [lid.strip() for lid in args.lesion_ids.split(",")]
    else:
        with retriever.driver.session() as s:
            rows = s.run(
                "MATCH (l:Lesion) WHERE l.embedding IS NOT NULL "
                "RETURN l.id AS id LIMIT $n", n=args.n_samples * 3,
            ).data()
        all_ids = [r["id"] for r in rows if r["id"]]
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_ids)
        sample_ids = all_ids[:args.n_samples]

    print(f"  Evaluating {len(sample_ids)} lesions:")
    for lid in sample_ids:
        print(f"    • {lid}")
    print()

    # ── Run evaluations ──
    results = []
    for i, lid in enumerate(sample_ids, 1):
        print(f"  [{i}/{len(sample_ids)}] {lid} ...", end=" ", flush=True)
        t_start = time.time()

        result = evaluate_single_query(
            lesion_id=lid,
            retriever=retriever,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        results.append(result)

        elapsed = time.time() - t_start
        status = result["status"]
        rag = result.get("rag_metrics", {})

        if status == "success":
            faith = rag.get("faithfulness", {}).get("score", "?")
            ground = rag.get("grounding", {}).get("score", "?")
            halluc = rag.get("hallucination", {}).get("hallucination_rate", "?")
            conf = result.get("confidence", {}).get("label", "?")
            print(f"✅ {elapsed:.1f}s | faith={faith} ground={ground} halluc={halluc} conf={conf}")
        else:
            print(f"❌ {status} ({elapsed:.1f}s)")

    retriever.close()

    # ── Aggregate ──
    print()
    print("=" * 60)
    print("  AGGREGATE RESULTS")
    print("=" * 60)

    summary = aggregate_results(results)
    summary["config"] = {
        "n_samples": len(sample_ids),
        "top_k": args.top_k,
        "min_score": args.min_score,
        "model": OLLAMA_MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Print summary
    print(f"\n  Success rate:           {summary['success_rate']:.0%} ({summary['n_successful']}/{summary['n_total']})")

    print(f"\n  ── Guardrail Pass Rates ──")
    for k, v in summary.get("guardrail_rates", {}).items():
        print(f"    {k}: {v:.0%}")

    print(f"\n  ── Retrieval ──")
    ret = summary.get("retrieval", {})
    print(f"    Mean top score:       {ret.get('mean_top_score', 0):.4f} ± {ret.get('std_top_score', 0):.4f}")
    print(f"    Score range:          [{ret.get('min_top_score', 0):.4f}, {ret.get('max_top_score', 0):.4f}]")

    print(f"\n  ── RAG Quality ──")
    rq = summary.get("rag_quality", {})
    print(f"    Structure pass rate:   {rq.get('structure_pass_rate', 0):.0%}")
    print(f"    Bullet format pass:    {rq.get('bullet_format_pass_rate', 0):.0%}")
    print(f"    Numeric accuracy:      {rq.get('numeric_accuracy_rate', 0):.0%}")
    print(f"    Faithfulness (mean):   {rq.get('faithfulness_mean', 0):.4f} ± {rq.get('faithfulness_std', 0):.4f}")
    print(f"    Grounding (mean):      {rq.get('grounding_mean', 0):.4f} ± {rq.get('grounding_std', 0):.4f}")
    print(f"    Hallucination rate:    {rq.get('hallucination_rate_mean', 0):.4f}")
    print(f"    Completeness (mean):   {rq.get('completeness_mean', 0):.4f}")
    print(f"    Uncertainty valid:     {rq.get('uncertainty_valid_rate', 0):.0%}")

    print(f"\n  ── Confidence Calibration ──")
    cc = summary.get("confidence_calibration", {})
    print(f"    Brier Score:           {cc.get('brier_score', 0):.4f} ({cc.get('brier_interpretation', '?')})")
    print(f"    Distribution:          {cc.get('confidence_distribution', {})}")
    print(f"    Outcome rate:          {cc.get('outcome_rate', 0):.0%}")

    print(f"\n  ── Latency ──")
    lt = summary.get("latency", {})
    print(f"    Retrieval:             {lt.get('retrieval_mean_s', 0):.3f}s ± {lt.get('retrieval_std_s', 0):.3f}s")
    print(f"    LLM:                   {lt.get('llm_mean_s', 0):.3f}s ± {lt.get('llm_std_s', 0):.3f}s")
    print(f"    Total (mean):          {lt.get('total_mean_s', 0):.3f}s (max={lt.get('total_max_s', 0):.3f}s)")

    # ── Save ──
    (args.out_dir / "rag_eval_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (args.out_dir / "rag_eval_per_query.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"\n  ✅ Results saved to: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
