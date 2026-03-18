#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for MedRAG-X:
# - Neo4j (Docker)
# - Ollama service + model
# - FastAPI backend
# - Streamlit chat app
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Optional overrides:
#   NEO4J_PASSWORD=neo4j123 OLLAMA_MODEL=llama3.1:8b ./run_all.sh
#   MEDRAGX_EMBEDDER_BACKEND=hash ./run_all.sh
#   NEO4J_RESET_CONTAINER=1 ./run_all.sh   # destructive: recreates neo4j container

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# -----------------------------
# Config (override via env)
# -----------------------------
export NEO4J_CONTAINER_NAME="${NEO4J_CONTAINER_NAME:-neo4j}"
export NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j123}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434/api/chat}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
export MEDRAGX_EMBEDDER_BACKEND="${MEDRAGX_EMBEDDER_BACKEND:-hash}"
export KG_JSON_PATH="${KG_JSON_PATH:-artifacts/kg_3d.json}"
export EMBEDDINGS_JSONL_PATH="${EMBEDDINGS_JSONL_PATH:-artifacts/embeddings/lesion_embeddings.jsonl}"

# Ethical guardrails defaults
export GUARDRAIL_ALLOWED_PURPOSES="${GUARDRAIL_ALLOWED_PURPOSES:-clinical_reasoning,research,education,quality_assurance}"
export GUARDRAIL_ALLOWED_LLM_HOSTS="${GUARDRAIL_ALLOWED_LLM_HOSTS:-127.0.0.1,localhost}"
export GUARDRAIL_ALLOWED_MODELS="${GUARDRAIL_ALLOWED_MODELS:-$OLLAMA_MODEL}"
export GUARDRAIL_STRICT_VENDOR_CHECK="${GUARDRAIL_STRICT_VENDOR_CHECK:-true}"
export GUARDRAIL_MAX_QUERY_CHARS="${GUARDRAIL_MAX_QUERY_CHARS:-1200}"
export GUARDRAIL_RETENTION_DAYS="${GUARDRAIL_RETENTION_DAYS:-30}"
export GUARDRAIL_UPLOAD_RETENTION_DAYS="${GUARDRAIL_UPLOAD_RETENTION_DAYS:-7}"
export GUARDRAIL_AUDIT_LOG="${GUARDRAIL_AUDIT_LOG:-artifacts/audit/audit_log.jsonl}"
export GUARDRAIL_UPLOADS_DIR="${GUARDRAIL_UPLOADS_DIR:-artifacts/uploads}"

# API defaults
export MEDRAGX_API_DEFAULT_ROLE="${MEDRAGX_API_DEFAULT_ROLE:-viewer}"
export PYTHONPATH="$ROOT_DIR"

mkdir -p artifacts/logs artifacts/audit artifacts/uploads

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

log() { echo "[run_all] $*"; }
warn() { echo "[run_all][warn] $*"; }
err() { echo "[run_all][error] $*" >&2; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required command not found: $1"
    exit 1
  fi
}

is_port_open() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY'
import socket, sys
h = sys.argv[1]
p = int(sys.argv[2])
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect((h, p))
    print("1")
except Exception:
    print("0")
finally:
    s.close()
PY
}

wait_for_http() {
  local url="$1"
  local max_attempts="${2:-60}"
  local i=0
  while [[ "$i" -lt "$max_attempts" ]]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    i=$((i + 1))
    sleep 1
  done
  return 1
}

ensure_knowledge_graph_artifacts() {
  require_cmd python3

  local kg_lesions="0"
  if [[ ! -s "$KG_JSON_PATH" ]]; then
    if ls data/processed/train/*.pt >/dev/null 2>&1; then
      log "KG file not found. Building Knowledge Graph from processed PT files..."
      python3 scripts/build_kg_3d_from_pt.py
    else
      warn "KG file missing ($KG_JSON_PATH) and no data/processed/train/*.pt found."
      warn "Skipping KG build."
    fi
  fi

  if [[ -s "$KG_JSON_PATH" ]]; then
    kg_lesions="$(
      python3 - "$KG_JSON_PATH" <<'PY'
import json, sys
p = sys.argv[1]
try:
    data = json.load(open(p, "r", encoding="utf-8"))
    nodes = data.get("nodes", [])
    cnt = sum(1 for n in nodes if n.get("type") == "Lesion" and isinstance(n.get("embedding"), list) and len(n.get("embedding")) > 0)
    print(cnt)
except Exception:
    print(0)
PY
    )"
    log "Knowledge Graph ready: $KG_JSON_PATH (lesions with embedding: ${kg_lesions:-0})"
  fi

  if [[ ! -s "$EMBEDDINGS_JSONL_PATH" && -s "$KG_JSON_PATH" ]]; then
    log "Embeddings JSONL missing. Generating from KG..."
    python3 scripts/build_embeddings_jsonl_from_kg.py --kg "$KG_JSON_PATH" --out "$EMBEDDINGS_JSONL_PATH"
  fi
}

start_neo4j() {
  require_cmd docker

  if ! docker info >/dev/null 2>&1; then
    err "Docker daemon is not running. Start Docker Desktop first."
    exit 1
  fi

  if [[ "${NEO4J_RESET_CONTAINER:-0}" == "1" ]]; then
    warn "NEO4J_RESET_CONTAINER=1: removing existing Neo4j container/data."
    docker rm -f "$NEO4J_CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  if docker ps -a --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER_NAME"; then
    log "Starting existing Neo4j container: $NEO4J_CONTAINER_NAME"
    docker start "$NEO4J_CONTAINER_NAME" >/dev/null || true
  else
    log "Creating Neo4j container: $NEO4J_CONTAINER_NAME"
    docker run --name "$NEO4J_CONTAINER_NAME" \
      -p 7474:7474 -p 7687:7687 \
      -e "NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASSWORD}" \
      -d neo4j:5 >/dev/null
  fi

  log "Waiting for Neo4j authentication..."
  local ok=0
  for _ in $(seq 1 90); do
    if docker exec "$NEO4J_CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 2
  done
  if [[ "$ok" -ne 1 ]]; then
    err "Neo4j did not become ready with provided credentials."
    err "If password is different from '$NEO4J_PASSWORD', set NEO4J_PASSWORD correctly."
    err "If you want reset, run: NEO4J_RESET_CONTAINER=1 ./run_all.sh"
    exit 1
  fi

  local lesions
  lesions="$(docker exec "$NEO4J_CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
    "MATCH (l:Lesion) RETURN count(l) AS lesions;" --format plain 2>/dev/null | tail -n 1 | tr -d '[:space:]' || echo "0")"
  log "Neo4j ready. Lesions in DB: ${lesions:-0}"

  if [[ "${lesions:-0}" == "0" ]]; then
    if [[ -s "$EMBEDDINGS_JSONL_PATH" ]]; then
      log "No lesions found. Ingesting embeddings into Neo4j..."
      python3 -m orchestration.pipeline run \
        --build_neo4j \
        --embeddings_jsonl "$EMBEDDINGS_JSONL_PATH" \
        --neo4j_password "$NEO4J_PASSWORD"
    else
      warn "No lesions in Neo4j and embeddings file missing/empty: $EMBEDDINGS_JSONL_PATH"
      warn "RAG comparison queries will fail until you build embeddings and ingest Neo4j."
    fi
  fi
}

start_ollama() {
  require_cmd ollama

  if ! wait_for_http "http://127.0.0.1:11434/api/tags" 2; then
    log "Starting Ollama server..."
    nohup ollama serve > artifacts/logs/ollama.log 2>&1 &
    if ! wait_for_http "http://127.0.0.1:11434/api/tags" 90; then
      err "Ollama server failed to start. Check: artifacts/logs/ollama.log"
      exit 1
    fi
  else
    log "Ollama server already running."
  fi

  log "Ensuring Ollama model is available: $OLLAMA_MODEL"
  ollama pull "$OLLAMA_MODEL" >/dev/null
}

start_api() {
  require_cmd python3

  if [[ "$(is_port_open 127.0.0.1 8000)" == "1" ]]; then
    log "API already running on :8000"
    return
  fi

  log "Starting FastAPI on :8000"
  nohup python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000 \
    > artifacts/logs/api.log 2>&1 &

  if ! wait_for_http "http://127.0.0.1:8000/docs" 60; then
    err "API failed to start. Check: artifacts/logs/api.log"
    exit 1
  fi
}

start_streamlit() {
  require_cmd streamlit

  # If streamlit is already running, do not start another.
  if [[ "$(is_port_open 127.0.0.1 8501)" == "1" ]]; then
    log "Streamlit already running on :8501"
    echo
    echo "Ready:"
    echo "  - Streamlit: http://127.0.0.1:8501"
    echo "  - API:       http://127.0.0.1:8000/docs"
    echo "  - Neo4j:     http://127.0.0.1:7474"
    return
  fi

  log "Starting Streamlit on :8501 (foreground)"
  echo
  echo "Ready:"
  echo "  - Streamlit: http://127.0.0.1:8501"
  echo "  - API:       http://127.0.0.1:8000/docs"
  echo "  - Neo4j:     http://127.0.0.1:7474"
  echo
  echo "Logs:"
  echo "  - API:       artifacts/logs/api.log"
  echo "  - Ollama:    artifacts/logs/ollama.log"
  echo
  exec streamlit run app/app.py
}

main() {
  log "Bootstrapping MedRAG-X stack..."
  ensure_knowledge_graph_artifacts
  start_neo4j
  start_ollama
  start_api
  start_streamlit
}

main "$@"
