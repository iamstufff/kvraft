#!/usr/bin/env bash
# run-local-cluster.sh — spin up three kvraft uvicorn processes on the
# local box with real pysyncobj Raft wiring. No Docker needed.
#
# HTTP ports: 8001, 8002, 8003
# Raft ports: 4321, 4322, 4323
#
# PIDs and logs land under /tmp/kvraft-local/.
# Use scripts/kill-local-leader.sh to take down the current leader.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR=/tmp/kvraft-local
mkdir -p "${STATE_DIR}"

: "${GEMINI_API_KEY:?set GEMINI_API_KEY in your env or .env}"

start_node() {
  local id="$1" http_port="$2" raft_port="$3" peers="$4"
  local logfile="${STATE_DIR}/${id}.log"
  local pidfile="${STATE_DIR}/${id}.pid"

  echo "[*] starting ${id} on :${http_port} (raft :${raft_port})"
  (
    export NODE_ID="${id}"
    export RAFT_BIND="127.0.0.1:${raft_port}"
    export RAFT_PEERS="${peers}"
    export SIMILARITY_THRESHOLD=0.85
    export GEMINI_API_KEY
    nohup "${ROOT}/.venv/bin/uvicorn" src.api:app \
      --host 127.0.0.1 --port "${http_port}" \
      >"${logfile}" 2>&1 &
    echo $! >"${pidfile}"
  )
}

start_node node-1 8001 4321 '["127.0.0.1:4322","127.0.0.1:4323"]'
start_node node-2 8002 4322 '["127.0.0.1:4321","127.0.0.1:4323"]'
start_node node-3 8003 4323 '["127.0.0.1:4321","127.0.0.1:4322"]'

echo "[*] waiting for /health on all three nodes..."
for port in 8001 8002 8003; do
  until curl -fs "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    sleep 1
  done
  echo "[+] :${port} ready"
done

echo "[+] cluster up. PIDs:"
for id in node-1 node-2 node-3; do
  echo "    ${id} → $(cat "${STATE_DIR}/${id}.pid")"
done
