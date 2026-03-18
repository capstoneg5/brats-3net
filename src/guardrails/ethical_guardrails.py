from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Set, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class EthicalDecision:
    allowed: bool
    code: str
    reason: str


@dataclass(frozen=True)
class EthicalPolicyConfig:
    allowed_purposes: Set[str]
    role_permissions: Dict[str, Set[str]]
    allowed_llm_hosts: Set[str]
    allowed_llm_models: Set[str]
    strict_vendor_check: bool
    audit_log_path: Path
    uploads_dir: Path
    max_query_chars: int
    max_audit_preview_chars: int
    retention_days: int
    upload_retention_days: int
    deny_alert_threshold: int
    deny_alert_window_minutes: int


DEFAULT_PURPOSES = {"clinical_reasoning", "research", "education", "quality_assurance"}
DEFAULT_ROLE_PERMISSIONS = {
    "viewer": {"chat", "rag_query", "search", "health"},
    "analyst": {"chat", "rag_query", "search", "embed_text", "health", "view_raw_context"},
    "admin": {"chat", "rag_query", "search", "embed_text", "health", "view_raw_context", "export_data", "policy_admin"},
}
DEFAULT_ALLOWED_LLM_HOSTS = {"127.0.0.1", "localhost"}

_SECRET_KEYWORDS = ("password", "secret", "token", "api_key", "apikey", "authorization", "bearer")
_RISKY_KEYWORDS = (
    "export",
    "share",
    "download all",
    "dump",
    "raw context",
    "full context",
    "all patient",
    "all lesions",
)

_SENSITIVE_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]*)?\(?\d{3}\)?[-.\s]*\d{3}[-.\s]*\d{4}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
]
_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(password|secret|token|api[_-]?key|authorization)\b\s*[:=]\s*([^\s,;]+)"
)


def _env_set(name: str, default: Set[str]) -> Set[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set(default)
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _env_int(name: str, default: int, min_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_ethical_policy() -> EthicalPolicyConfig:
    allowed_models = _env_set("GUARDRAIL_ALLOWED_MODELS", {os.getenv("OLLAMA_MODEL", "llama3.1:8b").lower()})
    return EthicalPolicyConfig(
        allowed_purposes=_env_set("GUARDRAIL_ALLOWED_PURPOSES", DEFAULT_PURPOSES),
        role_permissions={k: set(v) for k, v in DEFAULT_ROLE_PERMISSIONS.items()},
        allowed_llm_hosts=_env_set("GUARDRAIL_ALLOWED_LLM_HOSTS", DEFAULT_ALLOWED_LLM_HOSTS),
        allowed_llm_models=allowed_models,
        strict_vendor_check=_env_bool("GUARDRAIL_STRICT_VENDOR_CHECK", True),
        audit_log_path=Path(os.getenv("GUARDRAIL_AUDIT_LOG", "artifacts/audit/audit_log.jsonl")),
        uploads_dir=Path(os.getenv("GUARDRAIL_UPLOADS_DIR", "artifacts/uploads")),
        max_query_chars=_env_int("GUARDRAIL_MAX_QUERY_CHARS", 1200, 100),
        max_audit_preview_chars=_env_int("GUARDRAIL_MAX_AUDIT_PREVIEW_CHARS", 180, 40),
        retention_days=_env_int("GUARDRAIL_RETENTION_DAYS", 30, 1),
        upload_retention_days=_env_int("GUARDRAIL_UPLOAD_RETENTION_DAYS", 7, 1),
        deny_alert_threshold=_env_int("GUARDRAIL_DENY_ALERT_THRESHOLD", 4, 2),
        deny_alert_window_minutes=_env_int("GUARDRAIL_DENY_ALERT_WINDOW_MINUTES", 10, 1),
    )


def redact_sensitive_text(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False

    redacted = text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    redacted = _INLINE_SECRET_PATTERN.sub(r"\1=[REDACTED]", redacted)

    changed = redacted != text
    return redacted, changed


def minimize_query_text(query: str, policy: Optional[EthicalPolicyConfig] = None) -> Tuple[str, Dict[str, Any]]:
    policy = policy or get_ethical_policy()
    normalized = (query or "").strip()
    truncated = False
    if len(normalized) > policy.max_query_chars:
        normalized = normalized[: policy.max_query_chars]
        truncated = True

    redacted, redacted_any = redact_sensitive_text(normalized)
    return redacted, {
        "truncated": truncated,
        "redacted": redacted_any,
        "length": len(redacted),
        "sha256": hashlib.sha256(redacted.encode("utf-8")).hexdigest(),
    }


def check_purpose_allowed(purpose: str, policy: Optional[EthicalPolicyConfig] = None) -> EthicalDecision:
    policy = policy or get_ethical_policy()
    normalized = (purpose or "").strip().lower()
    if not normalized:
        return EthicalDecision(False, "missing_purpose", "Missing declared purpose.")
    if normalized not in policy.allowed_purposes:
        return EthicalDecision(False, "purpose_not_allowed", f"Purpose '{normalized}' is not approved.")
    return EthicalDecision(True, "allowed", "Purpose approved.")


def check_role_permission(role: str, action: str, policy: Optional[EthicalPolicyConfig] = None) -> EthicalDecision:
    policy = policy or get_ethical_policy()
    normalized_role = (role or "").strip().lower() or "viewer"
    allowed_actions = policy.role_permissions.get(normalized_role)
    if allowed_actions is None:
        return EthicalDecision(False, "unknown_role", f"Role '{normalized_role}' is not recognized.")
    if action not in allowed_actions:
        return EthicalDecision(False, "permission_denied", f"Role '{normalized_role}' cannot perform '{action}'.")
    return EthicalDecision(True, "allowed", "Action authorized.")


def _matches_allowlist(value: str, allowlist: Set[str]) -> bool:
    if not allowlist:
        return True
    candidate = value.strip().lower()
    for pattern in allowlist:
        if fnmatch.fnmatch(candidate, pattern):
            return True
    return False


def check_vendor_governance(llm_url: str, model: str, policy: Optional[EthicalPolicyConfig] = None) -> EthicalDecision:
    policy = policy or get_ethical_policy()
    if not policy.strict_vendor_check:
        return EthicalDecision(True, "allowed", "Vendor checks disabled by policy.")

    host = (urlparse(llm_url).hostname or "").lower()
    if not _matches_allowlist(host, policy.allowed_llm_hosts):
        return EthicalDecision(False, "vendor_blocked", f"LLM host '{host}' is not allowlisted.")

    normalized_model = (model or "").strip().lower()
    if not _matches_allowlist(normalized_model, policy.allowed_llm_models):
        return EthicalDecision(False, "model_blocked", f"Model '{model}' is not allowlisted.")

    return EthicalDecision(True, "allowed", "Vendor and model approved.")


def detect_risky_action(text: str) -> Dict[str, Any]:
    normalized = (text or "").strip().lower()
    reasons = []

    if any(keyword in normalized for keyword in _RISKY_KEYWORDS):
        reasons.append("high_impact_data_action")

    has_secret_keyword = any(k in normalized for k in _SECRET_KEYWORDS)
    _, redacted = redact_sensitive_text(normalized)
    if has_secret_keyword or redacted:
        reasons.append("sensitive_data_present")

    return {"risky": bool(reasons), "reasons": reasons}


def check_hitl_approval(text: str, approved: bool) -> EthicalDecision:
    risk = detect_risky_action(text)
    if risk["risky"] and not approved:
        return EthicalDecision(
            False,
            "hitl_required",
            f"Human approval required before risky action: {', '.join(risk['reasons'])}.",
        )
    return EthicalDecision(True, "allowed", "HITL check passed.")


def sanitize_for_audit(value: Any, max_len: int) -> Any:
    if isinstance(value, str):
        redacted, _ = redact_sensitive_text(value)
        return redacted[:max_len]
    if isinstance(value, Mapping):
        out: MutableMapping[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = sanitize_for_audit(v, max_len)
        return dict(out)
    if isinstance(value, list):
        return [sanitize_for_audit(v, max_len) for v in value[:30]]
    return value


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _core_event_payload(event: Dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _append_event_with_chain(log_path: Path, event_core: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prev_hash = "GENESIS"
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    last = json.loads(line)
                    prev_hash = str(last.get("entry_hash", prev_hash))
        except Exception:
            prev_hash = "GENESIS"

    payload = _core_event_payload(event_core)
    entry_hash = hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()
    event = dict(event_core)
    event["prev_hash"] = prev_hash
    event["entry_hash"] = entry_hash
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def _count_recent_denies(log_path: Path, actor: str, minutes: int) -> int:
    if not log_path.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    count = 0
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                evt = json.loads(line)
                if evt.get("outcome") != "deny":
                    continue
                if evt.get("actor") != actor:
                    continue
                ts = _parse_iso_utc(str(evt.get("ts", "")))
                if ts and ts >= cutoff:
                    count += 1
    except Exception:
        return 0
    return count


def append_audit_event(
    event_type: str,
    outcome: str,
    actor: str,
    role: str,
    purpose: str,
    details: Optional[Dict[str, Any]] = None,
    policy: Optional[EthicalPolicyConfig] = None,
) -> None:
    policy = policy or get_ethical_policy()
    safe_details = sanitize_for_audit(details or {}, policy.max_audit_preview_chars)
    core = {
        "ts": _now_utc_iso(),
        "event_type": event_type,
        "outcome": outcome,
        "actor": (actor or "unknown")[:80],
        "role": (role or "unknown")[:40],
        "purpose": (purpose or "unknown")[:80],
        "details": safe_details,
    }
    _append_event_with_chain(policy.audit_log_path, core)

    if outcome == "deny":
        deny_count = _count_recent_denies(
            policy.audit_log_path,
            actor=(actor or "unknown")[:80],
            minutes=policy.deny_alert_window_minutes,
        )
        if deny_count == policy.deny_alert_threshold:
            alert_core = {
                "ts": _now_utc_iso(),
                "event_type": "security_alert",
                "outcome": "alert",
                "actor": (actor or "unknown")[:80],
                "role": (role or "unknown")[:40],
                "purpose": (purpose or "unknown")[:80],
                "details": {
                    "reason": "repeated_denied_attempts",
                    "window_minutes": policy.deny_alert_window_minutes,
                    "deny_count": deny_count,
                },
            }
            _append_event_with_chain(policy.audit_log_path, alert_core)


def prune_uploads(uploads_dir: Path, older_than_days: int) -> int:
    if not uploads_dir.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    removed = 0
    for path in uploads_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                path.unlink()
                removed += 1
        except Exception:
            continue
    return removed


def prune_audit_log(log_path: Path, older_than_days: int) -> int:
    if not log_path.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    kept = []
    removed = 0

    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                evt = json.loads(line)
                ts = _parse_iso_utc(str(evt.get("ts", "")))
                if ts and ts < cutoff:
                    removed += 1
                    continue
                kept.append(evt)
    except Exception:
        return 0

    prev_hash = "GENESIS"
    rewritten = []
    for evt in kept:
        core = {k: v for k, v in evt.items() if k not in {"prev_hash", "entry_hash"}}
        payload = _core_event_payload(core)
        entry_hash = hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()
        core["prev_hash"] = prev_hash
        core["entry_hash"] = entry_hash
        rewritten.append(core)
        prev_hash = entry_hash

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for evt in rewritten:
            f.write(json.dumps(evt, ensure_ascii=True) + "\n")

    return removed


def enforce_retention_policy(policy: Optional[EthicalPolicyConfig] = None) -> Dict[str, int]:
    policy = policy or get_ethical_policy()
    uploads_removed = prune_uploads(policy.uploads_dir, policy.upload_retention_days)
    audit_removed = prune_audit_log(policy.audit_log_path, policy.retention_days)
    return {"uploads_removed": uploads_removed, "audit_entries_removed": audit_removed}
