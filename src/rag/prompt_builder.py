from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EvidenceItem:
    rank: int
    lesion_id: str
    patient_id: str
    score: float
    facts: Dict[str, Any]


class PromptBuilder:
    def __init__(self, few_shots: List[Dict[str, str]] | None = None):
        self.few_shots = few_shots or []

    def build(self, query: str, retrieved: List[EvidenceItem], safety_rules: List[str]) -> str:
        shots = "\n\n".join(
            f"User: {s['user']}\nAssistant: {s['assistant']}" for s in self.few_shots
        )

        evidence = "\n".join(
            f"[{e.rank}] lesion={e.lesion_id} patient={e.patient_id} score={e.score:.4f}"
            for e in retrieved
        )

        safety = " ".join(safety_rules)

        return f"""
You are a clinical research assistant. Use ONLY provided evidence.

{shots}

QUERY:
{query}

EVIDENCE:
{evidence}

SAFETY RULES:
{safety}

Provide concise clinical comparison.
"""