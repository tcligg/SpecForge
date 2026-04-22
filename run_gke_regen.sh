#!/usr/bin/env bash
#
# Thin wrapper around gke/orchestrator.py. Step 4a of gke/PLAN.md.
#
# The previous version of this script invoked gke/deploy.py --execute
# directly. The orchestrator now owns the entrypoint; Phase A (prepare)
# runs natively here, and Phases B-E delegate to deploy.py until steps
# 4b/6 land.
#
# Usage:
#   bash run_gke_regen.sh <config.yaml> [datasets] [gs://output-bucket/path] [extra orchestrator args...]
#
# Example:
#   bash run_gke_regen.sh gke/regen-gemma3-27b.yaml \
#       perfectblend,magpie-multilingual gs://my-bucket/output
#
# Resume an existing run (pass empty placeholders to keep positional args aligned):
#   bash run_gke_regen.sh gke/regen-gemma3-27b.yaml "" "" --run-id <id>

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ORCH="${SCRIPT_DIR}/gke/orchestrator.py"

if [ ! -f "${ORCH}" ]; then
    echo "Error: orchestrator not found at ${ORCH}" >&2
    exit 1
fi

JOB_YAML="${1:-}"
DATASETS="${2:-}"
OUTPUT_DIR="${3:-}"
[ "$#" -ge 1 ] && shift
[ "$#" -ge 1 ] && shift
[ "$#" -ge 1 ] && shift

if [ -z "${JOB_YAML}" ]; then
    echo "Usage: $0 <config.yaml> [datasets] [gs://output] [--run-id ID ...]" >&2
    exit 1
fi

ARGS=(run --config "${JOB_YAML}")
[ -n "${DATASETS}" ] && ARGS+=(--datasets "${DATASETS}")
[ -n "${OUTPUT_DIR}" ] && ARGS+=(--output-dir "${OUTPUT_DIR}")

exec python3 "${ORCH}" "${ARGS[@]}" "$@"
