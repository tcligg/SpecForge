#!/usr/bin/env bash
#
# One-command script to run the entire SpecForge data regeneration process on GKE.
#
# This script orchestrates the following steps by calling gke/deploy.py:
#   1. Builds the Docker image and pushes it to Artifact Registry.
#   2. Deletes any old jobs for the target datasets.
#   3. Submits new regeneration jobs across multiple GKE clusters and GPU types.
#   4. Waits for one job per dataset to acquire resources and start running.
#   5. Monitors the winning jobs until they complete.
#   6. Merges the chunked output files on GCS into a final, single file.
#
# Usage:
#   bash run_gke_regen.sh [path/to/job-config.yaml] [dataset1,dataset2,...] [gs://output-bucket/path]
#
# Example:
#   bash run_gke_regen.sh gke/regen-gemma4-26b.yaml ultrachat,perfectblend gs://my-bucket/output
#
# If arguments are omitted, they will be prompted for interactively.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEPLOY_SCRIPT="${SCRIPT_DIR}/gke/deploy.py"

if [ ! -f "${DEPLOY_SCRIPT}" ]; then
    echo "Error: deploy.py not found at ${DEPLOY_SCRIPT}"
    exit 1
fi

JOB_YAML="${1:-}"
DATASETS="${2:-}"
OUTPUT_DIR="${3:-}"

# Interactive prompt if YAML is not provided
if [ -z "${JOB_YAML}" ]; then
    echo "Please select the job YAML to run:"
    YAMLS=($(find "${SCRIPT_DIR}/gke" -name "regen-*.yaml" -printf "%f
"))
    select y in "${YAMLS[@]}"; do
        if [ -n "$y" ]; then
            JOB_YAML="gke/${y}"
            break
        fi
    done
fi

# Interactive prompt if datasets are not provided
if [ -z "${DATASETS}" ]; then
    # Extract default datasets from the chosen YAML
    DEFAULT_DS=$(grep -A1 'name: DATASETS' "${JOB_YAML}" | grep 'value:' | sed 's/.*value: "\(.*\)"/\1/')
    read -p "Enter comma-separated datasets to regenerate [${DEFAULT_DS}]: " DATASETS_INPUT
    DATASETS="${DATASETS_INPUT:-${DEFAULT_DS}}"
fi

# Build the command arguments
CMD_ARGS=(--yaml "${JOB_YAML}" --datasets "${DATASETS}" --execute)
if [ -n "${OUTPUT_DIR}" ]; then
    CMD_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

echo ""
echo "============================================================"
echo "  Starting SpecForge GKE Regeneration"
echo "============================================================"
echo "  Job YAML: ${JOB_YAML}"
echo "  Datasets: ${DATASETS}"
if [ -n "${OUTPUT_DIR}" ]; then
    echo "  Output Dir: ${OUTPUT_DIR}"
fi
echo "============================================================"
echo ""

# Execute the end-to-end process
python3 "${DEPLOY_SCRIPT}" "${CMD_ARGS[@]}"

echo ""
echo "============================================================"
echo "  All tasks complete."
echo "============================================================"
echo ""

