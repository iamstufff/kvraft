#!/usr/bin/env bash
# kill-local-leader.sh — local-process variant of kill-leader.sh. Reads PIDs
# from /tmp/kvraft-local/ (written by run-local-cluster.sh), finds the node
# whose /metrics reports kvraft_leader_state=1, kill -9's it, and times how
# long the surviving pair take to elect a new leader.

set -euo pipefail

STATE_DIR=/tmp/kvraft-local
NODES=(node-1 node-2 node-3)
PORTS=(8001 8002 8003)

need() { command -v "$1" >/dev/null 2>&1 || { echo "need $1 on PATH" >&2; exit 1; }; }
need curl

is_leader() {
  local port="$1"
  local value
  value=$(curl -fs "http://127.0.0.1:${port}/metrics" 2>/dev/null \
    | awk '/^kvraft_leader_state / {print $2}' \
    | tail -1) || return 1
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
pidfile="${STATE_DIR}/${leader_name}.pid"
[[ -f "$pidfile" ]] || { echo "no pidfile at ${pidfile}" >&2; exit 1; }
leader_pid=$(cat "$pidfile")
echo "[*] leader: ${leader_name} (port ${leader_port}, pid ${leader_pid})"

echo "[*] killing pid ${leader_pid}..."
start_ns=$(date +%s%N)
kill -9 "${leader_pid}"

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
