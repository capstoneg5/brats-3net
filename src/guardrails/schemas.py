from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

Action = Literal["allow", "refuse", "needs_more_info", "rewrite"]

@dataclass
class GuardrailDecision:
    action: Action
    reason: str
    safe_reply: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalItem:
    source: str                       # "neo4j" | "kg_json" | "notes" etc.
    id: str                           # lesion id / doc id
    score: float
    facts: Dict[str, Any]             # structured facts you retrieved
    text: Optional[str] = None        # optional summary line for prompt

@dataclass
class RetrievalBundle:
    query: str
    items: List[RetrievalItem]
    k: int
    min_score: float
    warnings: List[str] = field(default_factory=list)

@dataclass
class PromptPack:
    system: str
    user: str
    context: str