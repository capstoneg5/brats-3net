from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    LESION_SIMILARITY = "lesion_similarity"
    PROGNOSIS = "prognosis"
    CLINICAL_QA = "clinical_qa"


@dataclass
class RoutedQuery:
    text: str
    qtype: QueryType
    lesion_id: str | None


def route_query(text: str) -> RoutedQuery:
    t = (text or "").lower()

    lesion_match = re.search(r"(lesion\d+)", t)
    lesion_id = lesion_match.group(1) if lesion_match else None

    if any(w in t for w in ["similar", "compare", "nearest"]):
        qtype = QueryType.LESION_SIMILARITY
    elif any(w in t for w in ["survival", "prognosis", "outcome"]):
        qtype = QueryType.PROGNOSIS
    else:
        qtype = QueryType.CLINICAL_QA

    return RoutedQuery(text=text, qtype=qtype, lesion_id=lesion_id)