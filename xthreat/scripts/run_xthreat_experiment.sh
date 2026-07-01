#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

mkdir -p tmp/logs tmp/data/skillcorner_xthreat

ts="$(date +%Y%m%d_%H%M%S)"
log_path="tmp/logs/xthreat_${ts}.log"
pid_path="tmp/data/skillcorner_xthreat/run.pid"

PYTHON_BIN="${PYTHON:-python}"

nohup bash -c '
  set -euo pipefail
  python_bin="$1"
  shift

  actions_path="tmp/data/skillcorner_xthreat/actions.parquet"
  train_out_dir="tmp/data/skillcorner_xthreat"
  action_args=()

  while (($# > 0)); do
    case "$1" in
      --out)
        if (($# < 2)); then
          echo "--out requires a path" >&2
          exit 2
        fi
        actions_path="$2"
        action_args+=("$1" "$2")
        shift 2
        ;;
      --out=*)
        actions_path="${1#*=}"
        action_args+=("$1")
        shift
        ;;
      --train-out-dir)
        if (($# < 2)); then
          echo "--train-out-dir requires a path" >&2
          exit 2
        fi
        train_out_dir="$2"
        shift 2
        ;;
      --train-out-dir=*)
        train_out_dir="${1#*=}"
        shift
        ;;
      *)
        action_args+=("$1")
        shift
        ;;
    esac
  done

  "${python_bin}" -m xthreat.skillcorner_actions "${action_args[@]}"
  "${python_bin}" -m xthreat.train_skillcorner_xthreat --actions "${actions_path}" --out-dir "${train_out_dir}"
' bash "${PYTHON_BIN}" "$@" \
  > "${log_path}" 2>&1 &

echo "$!" > "${pid_path}"
echo "SkillCorner xT workflow started"
echo "PID: $(cat "${pid_path}")"
echo "Log: ${log_path}"
echo "Tail: tail -f ${log_path}"
