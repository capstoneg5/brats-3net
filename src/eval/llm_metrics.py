# src/eval/llm_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import re

# Optional: nltk for sentence splitting (recommended)
try:
    from nltk.tokenize import sent_tokenize
except Exception:
    sent_tokenize = None

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Utilities
# -----------------------------
try:
    from nltk.tokenize import sent_tokenize
except Exception:
    sent_tokenize = None

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Try NLTK if available AND data exists
    if sent_tokenize is not None:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            # NLTK installed but tokenizer data missing → fallback
            pass

    # Dependency-free fallback (production-safe)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(1, -1)
    b = np.asarray(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


# -----------------------------
# Embedding backend
# -----------------------------
class Embedder:
    """
    Thin wrapper so you can swap sentence-transformers models later.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


# -----------------------------
# Metrics
# -----------------------------
def answer_correctness(
    embedder: Embedder,
    generated: str,
    reference: str,
) -> float:
    """
    Semantic similarity (cosine) between generated and reference answers.
    """
    if not generated.strip() or not reference.strip():
        return 0.0
    gen_emb = embedder.encode([generated])[0]
    ref_emb = embedder.encode([reference])[0]
    return clamp01(safe_cosine(gen_emb, ref_emb))

def _numeric_lexical_support(sent: str, docs: List[str]) -> bool:
    # Match lesion_id
    m = re.search(r"(lesion\d+)", sent.lower())
    lesion = m.group(1) if m else None

    # If lesion_id exists, require it in a doc
    if lesion and not any(lesion in d.lower() for d in docs):
        return False

    # If ET% present, require some ET% number presence in at least one doc
    # (simple but effective for your ET%/ΔET_pp table-style output)
    nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", sent)
    if nums:
        joined = " ".join(docs).lower()
        # if any number token appears (exact string match), accept
        if any(n in joined for n in nums):
            return True

    # If lesion matched but numbers not found, still accept (lesion-level grounding)
    return bool(lesion)


def evidence_grounding(embedder, answer: str, retrieved_docs: List[str], threshold: float = 0.75) -> Tuple[float, List[dict]]:
    sents = split_sentences(answer)
    if not sents:
        return 0.0, []

    supported = 0
    details = []

    doc_embs = embedder.encode(retrieved_docs)  # precompute once

    for s in sents:
        # 1) Numeric/lexical fallback first (handles your table-style outputs)
        if _numeric_lexical_support(s, retrieved_docs):
            supported += 1
            details.append({"sentence": s, "supported": True, "method": "lexical_numeric"})
            continue

        # 2) Otherwise use embedding similarity
        sent_emb = embedder.encode([s])
        sims = cosine_similarity(sent_emb, doc_embs)[0]
        ok = float(max(sims)) >= threshold
        supported += 1 if ok else 0
        details.append({"sentence": s, "supported": ok, "method": "embedding", "max_sim": float(max(sims))})

    return supported / len(sents), details


def faithfulness_hallucination_rate(
    embedder: Embedder,
    answer: str,
    retrieved_docs: List[str],
    grounding_threshold: float = 0.75,
) -> tuple[float, list[dict]]:
    """
    Hallucination rate = 1 - grounding score
    (i.e., fraction of sentences NOT supported by evidence)
    """
    grounding, details = evidence_grounding(embedder, answer, retrieved_docs, grounding_threshold)
    return clamp01(1.0 - grounding), details

def uncertainty_calibration(
    correctness: float,
    confidence_label: str,
) -> float:
    """
    Simple calibration score in [0,1]:
    - Map confidence label -> expected correctness range
    - Penalize mismatch between correctness and expected
    """
    label = (confidence_label or "").strip().lower()
    if label == "high":
        expected = 0.85
    elif label == "moderate":
        expected = 0.70
    else:
        expected = 0.50  # low confidence expects low correctness

    # Score is 1 when correctness close to expected, decreases with distance
    # (tunable)
    diff = abs(correctness - expected)
    score = 1.0 - min(diff / 0.50, 1.0)  # diff 0.5 => score 0
    return clamp01(score)


# -----------------------------
# End-to-end evaluation result
# -----------------------------
@dataclass
class RagEvalResult:
    correctness: float
    grounding: float
    hallucination_rate: float
    uncertainty_calibration: Optional[float]
    details: Dict[str, Any]


def evaluate_rag_answer(
    embedder: Embedder,
    generated_answer: str,
    retrieved_docs: List[str],
    reference_answer: Optional[str] = None,
    grounding_threshold: float = 0.75,
    confidence_label: Optional[str] = None,
) -> RagEvalResult:
    corr = None
    if reference_answer is not None:
        corr = answer_correctness(embedder, generated_answer, reference_answer)

    grounding, g_details = evidence_grounding(embedder, generated_answer, retrieved_docs, grounding_threshold)
    halluc, h_details = faithfulness_hallucination_rate(embedder, generated_answer, retrieved_docs, grounding_threshold)

    calib = None
    if (corr is not None) and confidence_label:
        calib = uncertainty_calibration(corr, confidence_label)

    return RagEvalResult(
        correctness=float(corr) if corr is not None else -1.0,
        grounding=float(grounding),
        hallucination_rate=float(halluc),
        uncertainty_calibration=float(calib) if calib is not None else None,
        details={
            "grounding_details": g_details,
            "hallucination_details": h_details,
            "grounding_threshold": grounding_threshold,
        },
    )