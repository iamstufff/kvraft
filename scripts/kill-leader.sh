#!/usr/bin/env bash
# kill-leader.sh — probe each kvraft container for Raft leadership, kill the
# leader, and time how long it takes for a new leader to be elected.
#
# Requirements: `docker` on PATH, `curl`, `jq`. Assumes the compose project
# defined in docker-compose.yml is up.

set -euo pipefail

NODES=(kvraft-1 kvraft-2 kvraft-3)
PORTS=(8001 8002 8003)

need() { command -v "$1" >/dev/null 2>&1 || { echo "need $1 on PATH" >&2; exit 1; }; }
need docker; need curl; need jq

is_leader() {
  local port="$1"
  # Leader-state gauge is exported on /metrics; 1.0 means this node thinks
  # it is the leader.
  local value
  value=$(curl -fs "http://127.0.0.1:${port}/metrics" \
    | awk '/^kvraft_leader_state / {print $2}' \
    | tail -1)
  [[ "$value" == "1" || "$value" == "1.0" ]]
}

find_leader() {
  for i in "${!NODES[@]}"; do
    if is_leader "${PORTS[$i]}"; then
      echo "${NODES[$i]}|${PORTS[$i]}|$i"
      return 0
    fi
  done
  return 1
}

wait_for_new_leader() {
  local killed_port="$1"
  local deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    for i in "${!NODES[@]}"; do
      [[ "${PORTS[$i]}" == "$killed_port" ]] && continue
      if is_leader "${PORTS[$i]}"; then
        echo "${NODES[$i]}|${PORTS[$i]}"
        return 0
      fi
    done
    sleep 0.1
  done
  return 1
}

echo "[*] locating current leader..."
leader_info=$(find_leader) || { echo "no leader reports kvraft_leader_state=1 — is the cluster up?" >&2; exit 1; }
IFS='|' read -r leader_name leader_port _ <<< "$leader_info"
echo "[*] leader: ${leader_name} (port ${leader_port})"

echo "[*] killing ${leader_name}..."
start_ns=$(date +%s%N)
docker kill "${leader_name}" >/dev/null

echo "[*] waiting for a new leader..."
if new_info=$(wait_for_new_leader "${leader_port}"); then
  end_ns=$(date +%s%N)
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
  IFS='|' read -r new_name new_port <<< "$new_info"
  echo "[+] new leader: ${new_name} (port ${new_port})"
  echo "[+] failover time: ${elapsed_ms} ms"
else
  echo "[-] no new leader within 30s" >&2
  exit 1
fi
