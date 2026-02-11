from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCK = "block"

@dataclass
class GuardrailDecision:
    allowed: bool
    risk: RiskLevel
    reasons: List[str]
    safe_response: Optional[str] = None
    prompt_rules: Optional[str] = None

# Simple policy phrases (expand anytime)
DIAGNOSIS_TRIGGERS = [
    "diagnose", "diagnosis", "what disease", "is it cancer", "is it tumor",
    "confirm", "definitely", "surely", "prove"
]

TREATMENT_TRIGGERS = [
    "treat", "treatment", "medicine", "drug", "dose", "prescribe", "chemotherapy",
    "radiation", "surgery", "should i", "what should i do", "recommend"
]

EMERGENCY_TRIGGERS = [
    "suicidal", "collapse", "unconscious", "seizure", "stroke", "emergency", "severe headache"
]

PRIVACY_TRIGGERS = [
    "phone", "address", "aadhar", "ssn", "email", "patient name", "dob"
]