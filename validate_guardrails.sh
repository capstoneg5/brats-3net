#!/usr/bin/env bash
set -u

# validate_guardrails.sh
# Automated checks for ethical guardrails in MedRAG-X API.
#
# Usage:
#   chmod +x validate_guardrails.sh
#   ./validate_guardrails.sh
#
# Optional env vars:
#   API_URL=http://127.0.0.1:8000
#   AUDIT_LOG=artifacts/audit/audit_log.jsonl
#   VALIDATE_API_KEY=<api-key>             # use when MEDRAGX_API_KEYS is enabled
#   EXPECT_PURPOSE_BLOCK=marketing

API_URL="${API_URL:-http://127.0.0.1:8000}"
AUDIT_LOG="${AUDIT_LOG:-artifacts/audit/audit_log.jsonl}"
EXPECT_PURPOSE_BLOCK="${EXPECT_PURPOSE_BLOCK:-marketing}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-45}"

PASS_COUNT=0
FAIL_COUNT=0

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

print_header() {
  printf "\n== %s ==\n" "$1"
}

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf "PASS: %s\n" "$1"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf "FAIL: %s\n" "$1"
}

request() {
  # request METHOD URL BODY OUTFILE HEADER...
  local method="$1"
  local url="$2"
  local body="$3"
  local outfile="$4"
  shift 4

  local -a args
  args=(
    -sS
    --connect-timeout "$CURL_CONNECT_TIMEOUT"
    --max-time "$CURL_MAX_TIME"
    -o "$outfile"
    -w "%{http_code}"
    -X "$method"
    "$url"
  )
  if [[ -n "$body" ]]; then
    args+=(-H "Content-Type: application/json" -d "$body")
  fi
  while [[ $# -gt 0 ]]; do
    args+=(-H "$1")
    shift
  done

  curl "${args[@]}"
}

is_api_down_status() {
  [[ "$1" == "000" ]]
}

request_auth() {
  # request_auth METHOD URL BODY OUTFILE ROLE PURPOSE HITL
  local method="$1"
  local url="$2"
  local body="$3"
  local outfile="$4"
  local role="$5"
  local purpose="$6"
  local hitl="${7:-}"

  if [[ -n "${VALIDATE_API_KEY:-}" ]]; then
    if [[ -n "$hitl" ]]; then
      request "$method" "$url" "$body" "$outfile" \
        "X-API-Key: ${VALIDATE_API_KEY}" \
        "X-Purpose: ${purpose}" \
        "X-HITL-Approved: ${hitl}"
    else
      request "$method" "$url" "$body" "$outfile" \
        "X-API-Key: ${VALIDATE_API_KEY}" \
        "X-Purpose: ${purpose}"
    fi
  else
    if [[ -n "$hitl" ]]; then
      request "$method" "$url" "$body" "$outfile" \
        "X-Role: ${role}" \
        "X-Purpose: ${purpose}" \
        "X-HITL-Approved: ${hitl}"
    else
      request "$method" "$url" "$body" "$outfile" \
        "X-Role: ${role}" \
        "X-Purpose: ${purpose}"
    fi
  fi
}

check_api_reachable() {
  local out="$TMP_DIR/reachable.txt"
  local status
  status="$(curl -sS -o "$out" -w "%{http_code}" "$API_URL/docs" || true)"
  if [[ "$status" == "200" || "$status" == "307" || "$status" == "308" ]]; then
    pass "API reachable at $API_URL"
    return 0
  else
    fail "API not reachable at $API_URL (status=$status). Start API: uvicorn app.api:app --reload --port 8000"
    return 1
  fi
}

test_purpose_limitation() {
  print_header "Purpose Limitation"
  local out="$TMP_DIR/purpose.json"
  local status
  status="$(request_auth GET "$API_URL/health" "" "$out" "viewer" "$EXPECT_PURPOSE_BLOCK" || true)"
  if [[ "$status" == "403" ]]; then
    pass "Blocked disallowed purpose ($EXPECT_PURPOSE_BLOCK)"
  else
    fail "Expected 403 for disallowed purpose, got $status"
  fi
}

test_least_privilege() {
  print_header "Least Privilege"
  local out1="$TMP_DIR/embed_viewer.json"
  local out2="$TMP_DIR/embed_analyst.json"

  local status1 status2
  status1="$(request_auth POST "$API_URL/embed/text" '{"texts":["brain tumor mri"]}' "$out1" "viewer" "research" || true)"
  if is_api_down_status "$status1"; then
    fail "API became unavailable during viewer embed_text check (status=000)"
    return
  fi
  if [[ "$status1" == "403" ]]; then
    pass "Viewer blocked from embed_text"
  else
    fail "Expected viewer embed_text to be 403, got $status1"
  fi

  status2="$(request_auth POST "$API_URL/embed/text" '{"texts":["brain tumor mri"]}' "$out2" "analyst" "research" || true)"
  if is_api_down_status "$status2"; then
    fail "API became unavailable during analyst embed_text check (status=000)"
  elif [[ "$status2" != "401" && "$status2" != "403" ]]; then
    pass "Analyst allowed past authorization on embed_text (status=$status2)"
  else
    fail "Analyst should pass authorization on embed_text, got $status2"
  fi
}

test_hitl_gate() {
  print_header "HITL Gate"
  local out1="$TMP_DIR/search_no_hitl.json"
  local out2="$TMP_DIR/search_hitl.json"
  local risky_query='{"query":"export raw context and share all patient data","top_k":3}'
  local status1 status2
  status1="$(request_auth POST "$API_URL/search" "$risky_query" "$out1" "analyst" "research" || true)"
  if is_api_down_status "$status1"; then
    fail "API became unavailable during HITL deny check (status=000)"
  elif [[ "$status1" == "403" ]]; then
    pass "Risky request blocked without HITL approval"
  else
    fail "Expected 403 without HITL approval, got $status1"
  fi

  status2="$(request_auth POST "$API_URL/search" "$risky_query" "$out2" "analyst" "research" "true" || true)"
  if is_api_down_status "$status2"; then
    fail "API became unavailable during HITL approved check (status=000)"
  elif [[ "$status2" != "401" && "$status2" != "403" ]]; then
    pass "Risky request passed HITL gate when approved (status=$status2)"
  else
    fail "Expected non-403 with HITL approval, got $status2"
  fi
}

test_audit_log_exists() {
  print_header "Audit Log"
  if [[ -f "$AUDIT_LOG" ]]; then
    pass "Audit log exists at $AUDIT_LOG"
  else
    fail "Audit log not found at $AUDIT_LOG"
  fi
}

test_audit_redaction() {
  local marker_email="guardrailtest.user@example.com"
  local marker_secret="password=my_secret_123"
  local out="$TMP_DIR/embed_redaction.json"
  # Trigger an audited event containing sensitive-looking text.
  request_auth POST "$API_URL/embed/text" "{\"texts\":[\"$marker_email $marker_secret\"]}" "$out" "analyst" "research" "true" >/dev/null 2>&1 || true

  if [[ ! -f "$AUDIT_LOG" ]]; then
    fail "Cannot test redaction; audit log does not exist"
    return
  fi

  if grep -q "$marker_email" "$AUDIT_LOG"; then
    fail "Audit log leaked raw email marker"
  else
    pass "Audit log did not leak raw email marker"
  fi

  if grep -q "my_secret_123" "$AUDIT_LOG"; then
    fail "Audit log leaked raw secret marker"
  else
    pass "Audit log did not leak raw secret marker"
  fi
}

test_hash_chain() {
  print_header "Audit Hash Chain"
  if [[ ! -f "$AUDIT_LOG" ]]; then
    fail "Cannot validate hash chain; audit log missing"
    return
  fi

  local out="$TMP_DIR/hash_chain.txt"
  if python3 - "$AUDIT_LOG" >"$out" 2>&1 <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
prev = "GENESIS"
with open(path, "r", encoding="utf-8") as fh:
    for idx, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        evt = json.loads(line)
        core = {k: v for k, v in evt.items() if k not in {"prev_hash", "entry_hash"}}
        payload = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        expected = hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()
        if evt.get("prev_hash") != prev or evt.get("entry_hash") != expected:
            print(f"BROKEN line={idx}")
            sys.exit(2)
        prev = evt["entry_hash"]
print("CHAIN_OK")
PY
  then
    if grep -q "CHAIN_OK" "$out"; then
      pass "Audit log hash chain is valid"
    else
      fail "Hash chain script succeeded but did not return CHAIN_OK"
    fi
  else
    fail "Audit hash chain invalid. $(cat "$out")"
  fi
}

summary() {
  print_header "Summary"
  printf "Passed: %d\n" "$PASS_COUNT"
  printf "Failed: %d\n" "$FAIL_COUNT"
  if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
  fi
}

print_header "MedRAG-X Guardrails Validation"
echo "API_URL=$API_URL"
echo "AUDIT_LOG=$AUDIT_LOG"
if [[ -n "${VALIDATE_API_KEY:-}" ]]; then
  echo "Auth mode: X-API-Key"
else
  echo "Auth mode: X-Role (no VALIDATE_API_KEY provided)"
fi

if ! check_api_reachable; then
  summary
fi

test_purpose_limitation
test_least_privilege
test_hitl_gate
test_audit_log_exists
test_audit_redaction
test_hash_chain
summary
