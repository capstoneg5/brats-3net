from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptBundle:
    system: str
    user: str
    context: str


def build_prompt(
    question: str,
    nifti_rag_text: Optional[str] = None,
    nifti_structured: Optional[Dict[str, Any]] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> PromptBundle:
    """
    Builds the final prompt for the LLM.
    This is intentionally deterministic to support clinical guardrails.
    """

    retrieved_chunks = retrieved_chunks or []

    system = (
        "You are MedRAG-X assistant.\n"
        "You must follow clinical safety rules.\n"
        "- Do NOT diagnose.\n"
        "- Do NOT prescribe treatment.\n"
        "- Report only computed imaging metrics when available.\n"
        "- If data is missing, say what is missing.\n"
    )

    parts: List[str] = []

    if nifti_rag_text:
        parts.append("### Imaging Case Summary\n" + nifti_rag_text)

    if nifti_structured:
        parts.append(
            "### Imaging Metrics (JSON)\n"
            + json.dumps(nifti_structured, indent=2)
        )

    if retrieved_chunks:
        lines = ["### Retrieved Knowledge"]
        for i, ch in enumerate(retrieved_chunks[:8], start=1):
            src = ch.get("source", f"chunk_{i}")
            txt = ch.get("text", "")
            score = ch.get("score", None)
            if score is not None:
                lines.append(f"[{i}] {src} (score={score})\n{txt}")
            else:
                lines.append(f"[{i}] {src}\n{txt}")
        parts.append("\n".join(lines))

    context = "\n\n".join(parts) if parts else "No additional context available."

    user = (
        f"Question: {question}\n\n"
        "Answer requirements:\n"
        "1) Report computed metrics if available (WT/TC/ET).\n"
        "2) Explain meaning at a high level (educational).\n"
        "3) Clearly state limitations.\n"
        "4) Do NOT give medical advice.\n"
    )

    return PromptBundle(system=system, user=user, context=context)