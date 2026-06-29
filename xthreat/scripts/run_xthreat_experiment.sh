#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

mkdir -p tmp/logs tmp/data/skillcorner_xthreat

ts="$(date +%Y%m%d_%H%M%S)"
log_path="tmp/logs/xthreat_${ts}.log"
pid_path="tmp/data/skillcorner_xthreat/run.pid"

nohup /data2/envs/mhl-py311-sky/bin/python -m xthreat.run_skillcorner_xthreat "$@" \
  > "${log_path}" 2>&1 &

echo "$!" > "${pid_path}"
echo "xThreat experiment started"
echo "PID: $(cat "${pid_path}")"
echo "Log: ${log_path}"
echo "Tail: tail -f ${log_path}"
