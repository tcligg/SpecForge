#!/usr/bin/env bash
# Deploy per-dataset regen batch jobs across multiple GKE clusters.
#
# For each dataset, the script:
#   1. Patches the YAML with that single dataset and a unique job name.
#   2. Submits to ALL clusters.
#   3. Waits until one cluster has all pods Running.
#   4. Deletes the job from all other clusters.
#   5. Moves on to the next dataset.
#
# Usage:
#   DATASETS=perfectblend,magpie-multilingual bash gke/deploy.sh
#   bash gke/deploy.sh                           # uses DATASETS from YAML
#   bash gke/deploy.sh --status                  # check all jobs
#   bash gke/deploy.sh --delete                  # delete all jobs
#
# Environment variables:
#   DATASETS          - Comma-separated dataset names (default: from YAML)
#   CLUSTERS          - Comma-separated "cluster:region" pairs (default: auto-discover)
#   PROJECT           - GCP project ID (default: cloud-llm-test)
#   SCHEDULE_TIMEOUT  - Seconds to wait per dataset for scheduling (default: 600)

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PROJECT="${PROJECT:-cloud-llm-test}"
SCHEDULE_TIMEOUT="${SCHEDULE_TIMEOUT:-3600}"
GPU_TYPES="${GPU_TYPES:-nvidia-h200-141gb,nvidia-h100-80gb}"
IFS=',' read -ra GPU_TYPE_ARRAY <<< "${GPU_TYPES}"
JOB_YAML="${JOB_YAML:-}"

if [ -z "${JOB_YAML}" ]; then
    echo "Error: JOB_YAML is required. Available YAMLs:"
    ls -1 "${SCRIPT_DIR}"/regen-*.yaml 2>/dev/null | sed 's/^/  /'
    echo ""
    echo "Usage: JOB_YAML=gke/regen-gemma4-26b.yaml bash gke/deploy.sh"
    exit 1
fi

if [ ! -f "${JOB_YAML}" ]; then
    echo "Error: ${JOB_YAML} not found."
    exit 1
fi

BASE_NAME=$(grep -m1 'name:' "${JOB_YAML}" | awk '{print $2}')
TOTAL_PODS=$(grep 'completions:' "${JOB_YAML}" | awk '{print $2}')
MIN_RUNNING=$(( (TOTAL_PODS + 1) / 2 ))  # at least half

DATASETS="${DATASETS:-magpie-multilingual,perfectblend}"

IFS=',' read -ra DATASET_ARRAY <<< "${DATASETS}"

# Discover clusters
if [ -z "${CLUSTERS:-}" ]; then
    echo "Discovering pyc-batch* clusters in project ${PROJECT}..."
    CLUSTERS=$(gcloud container clusters list \
        --project "${PROJECT}" \
        --filter="name~^pyc-batch AND NOT name=pyc-batch-us-south1" \
        --format="csv[no-heading](name,location)" \
    | tr ',' ':' \
    | paste -sd,)

    if [ -z "${CLUSTERS}" ]; then
        echo "Error: No pyc-batch* clusters found in project ${PROJECT}."
        exit 1
    fi
fi

IFS=',' read -ra CLUSTER_ARRAY <<< "${CLUSTERS}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

use_cluster() {
    local cluster="$1" region="$2"
    gcloud container clusters get-credentials "${cluster}" \
        --region "${region}" --project "${PROJECT}" \
        --quiet 2>/dev/null
}

get_pod_status_counts() {
    local job_name="$1"
    kubectl get pods -l "job-name=${job_name}" \
        --no-headers 2>/dev/null \
    | awk '
        { status[$3]++ }
        END {
            printf "%d %d %d",
                status["Running"]+0,
                status["Pending"]+0,
                (NR - status["Running"] - status["Pending"])+0
        }
    '
}

gpu_short_name() {
    # nvidia-h200-141gb -> h200, nvidia-h100-80gb -> h100
    echo "$1" | sed 's/nvidia-//; s/-.*//'
}

make_job_name() {
    # e.g. regen-gemma4-26b-perfectblend-h200-europe-west1
    local base="$1" dataset="$2" gpu="$3" region="$4"
    local gpu_short
    gpu_short=$(gpu_short_name "${gpu}")
    echo "${base}-${dataset}-${gpu_short}-${region}"
}

apply_patched_yaml() {
    local job_yaml="$1" base_name="$2" dataset="$3" gpu_type="$4" region="$5"
    local job_name
    job_name=$(make_job_name "${base_name}" "${dataset}" "${gpu_type}" "${region}")
    # Use awk to patch job name, DATASETS value, and GPU type.
    awk -v base="${base_name}" -v jname="${job_name}" -v ds="${dataset}" -v gpu="${gpu_type}" '
        /^  name:/ && !done_name { sub("name: "base, "name: "jname); done_name=1 }
        prev_datasets { sub(/value: ".*"/, "value: \""ds"\""); prev_datasets=0 }
        /name: DATASETS/ { prev_datasets=1 }
        /gke-accelerator:/ && !/accelerator-count/ { sub(/gke-accelerator: .*/, "gke-accelerator: "gpu) }
        { print }
    ' "${job_yaml}" \
    | kubectl apply -f -
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_status() {
    echo ""
    echo "Status for all dataset jobs:"
    echo "------------------------------------------------------------"
    for ds in "${DATASET_ARRAY[@]}"; do
        echo "  Dataset: ${ds}"
        for gpu in "${GPU_TYPE_ARRAY[@]}"; do
            for entry in "${CLUSTER_ARRAY[@]}"; do
                local cluster="${entry%%:*}" region="${entry##*:}"
                local job_name
                job_name=$(make_job_name "${BASE_NAME}" "${ds}" "${gpu}" "${region}")
                if use_cluster "${cluster}" "${region}"; then
                    local counts
                    counts=$(get_pod_status_counts "${job_name}")
                    local running pending other
                    read -r running pending other <<< "${counts}"
                    if [ $(( running + pending + other )) -eq 0 ]; then
                        continue  # no job here
                    fi
                    local job_status
                    job_status=$(kubectl get job "${job_name}" --no-headers 2>/dev/null \
                        | awk '{print $2}' || echo "not found")
                    echo "    ${job_name}: completions=${job_status}  pods: ${running} running, ${pending} pending, ${other} other"
                fi
            done
        done
    done
    echo ""
}

cmd_delete() {
    echo "Deleting all dataset jobs from all clusters..."
    for ds in "${DATASET_ARRAY[@]}"; do
        for gpu in "${GPU_TYPE_ARRAY[@]}"; do
            for entry in "${CLUSTER_ARRAY[@]}"; do
                local cluster="${entry%%:*}" region="${entry##*:}"
                local job_name
                job_name=$(make_job_name "${BASE_NAME}" "${ds}" "${gpu}" "${region}")
                if use_cluster "${cluster}" "${region}"; then
                    kubectl delete job "${job_name}" --ignore-not-found 2>&1 \
                        | grep -v "not found" \
                        | sed "s/^/  /" || true
                fi
            done
        done
    done
    echo "Done."
}

deploy_dataset() {
    local job_yaml="$1" base_name="$2" dataset="$3"

    echo ""
    echo "============================================================"
    echo "  Deploying: ${dataset}"
    echo "  GPU types: ${GPU_TYPES}"
    echo "============================================================"

    # ── Phase 1: Submit to all (cluster x gpu_type) combinations ─────
    # Each entry is "cluster:region:gpu_type"
    local submitted=()

    for gpu in "${GPU_TYPE_ARRAY[@]}"; do
        for entry in "${CLUSTER_ARRAY[@]}"; do
            local cluster="${entry%%:*}" region="${entry##*:}"
            local job_name
            job_name=$(make_job_name "${base_name}" "${dataset}" "${gpu}" "${region}")

            if ! use_cluster "${cluster}" "${region}"; then
                echo "  ${cluster}: failed to connect, skipping."
                continue
            fi

            local apply_err
            if apply_err=$(apply_patched_yaml "${job_yaml}" "${base_name}" "${dataset}" "${gpu}" "${region}" 2>&1); then
                echo "  ${job_name}: submitted."
                submitted+=("${entry}:${gpu}")
            else
                echo "  ${job_name}: kubectl apply failed, skipping."
                echo "    ${apply_err}" | head -5
            fi
        done
    done

    if [ ${#submitted[@]} -eq 0 ]; then
        echo "  Error: Failed to submit ${dataset} to any cluster."
        return 1
    fi

    # ── Phase 2: Wait for pods to schedule ───────────────────────────
    echo "  Waiting up to ${SCHEDULE_TIMEOUT}s for pods to schedule..."

    local elapsed=0
    local winner=""

    while [ ${elapsed} -lt ${SCHEDULE_TIMEOUT} ]; do
        for sub in "${submitted[@]}"; do
            # Parse "cluster:region:gpu_type"
            local cluster region gpu
            cluster=$(echo "${sub}" | cut -d: -f1)
            region=$(echo "${sub}" | cut -d: -f2)
            gpu=$(echo "${sub}" | cut -d: -f3-)
            local job_name
            job_name=$(make_job_name "${base_name}" "${dataset}" "${gpu}" "${region}")

            if ! use_cluster "${cluster}" "${region}"; then
                continue
            fi

            local counts
            counts=$(get_pod_status_counts "${job_name}")
            local running pending other
            read -r running pending other <<< "${counts}"

            if [ "${running}" -ge "${MIN_RUNNING}" ]; then
                winner="${sub}"
                break 2
            fi
        done

        echo "  [${elapsed}s] No fully scheduled cluster yet..."
        sleep 30
        elapsed=$(( elapsed + 30 ))
    done

    if [ -z "${winner}" ]; then
        echo "  Warning: ${dataset} not scheduled within ${SCHEDULE_TIMEOUT}s. Jobs remain submitted."
        return 0
    fi

    # ── Phase 3: Keep winner, delete the rest ────────────────────────
    local win_cluster win_region win_gpu winner_name
    win_cluster=$(echo "${winner}" | cut -d: -f1)
    win_region=$(echo "${winner}" | cut -d: -f2)
    win_gpu=$(echo "${winner}" | cut -d: -f3-)
    winner_name=$(make_job_name "${base_name}" "${dataset}" "${win_gpu}" "${win_region}")
    echo "  Scheduled on: ${winner_name}"

    for sub in "${submitted[@]}"; do
        if [ "${sub}" != "${winner}" ]; then
            local cluster region gpu job_name
            cluster=$(echo "${sub}" | cut -d: -f1)
            region=$(echo "${sub}" | cut -d: -f2)
            gpu=$(echo "${sub}" | cut -d: -f3-)
            job_name=$(make_job_name "${base_name}" "${dataset}" "${gpu}" "${region}")
            if use_cluster "${cluster}" "${region}"; then
                kubectl delete job "${job_name}" --ignore-not-found > /dev/null 2>&1 || true
                echo "  ${job_name}: deleted."
            fi
        fi
    done

    echo "  ${dataset} -> ${winner_name} (${win_cluster}, $(gpu_short_name "${win_gpu}"))"
}

cmd_build_and_redeploy() {
    # Build image, push, update YAML, delete old jobs, deploy new ones
    echo ""
    echo "Building and pushing new image..."
    local build_output
    build_output=$(bash "${SCRIPT_DIR}/build_and_push.sh" 2>&1)
    echo "${build_output}"

    # Extract the new tag from build output
    local new_tag
    new_tag=$(echo "${build_output}" | grep "Image:" | tail -1 | sed 's/.*specforge-regen://' | tr -d '[:space:]')
    if [ -z "${new_tag}" ]; then
        echo "Error: Could not extract image tag from build output."
        exit 1
    fi

    local new_image="us-central1-docker.pkg.dev/pyc-vtx-dev/pyc-vtx-us-central1/specforge-regen:${new_tag}"
    echo ""
    echo "Updating YAML image to: ${new_image}"

    # Update the image tag in the YAML — replace the entire image: line
    awk -v new="${new_image}" \
        '/^ *image:/ { sub(/image: .*/, "image: "new) } { print }' \
        "${JOB_YAML}" > "${JOB_YAML}.tmp" \
        && mv "${JOB_YAML}.tmp" "${JOB_YAML}"

    # Delete and redeploy
    echo ""
    cmd_delete
    echo ""
    cmd_deploy
}

cmd_redeploy() {
    # Delete existing jobs and redeploy with current YAML
    echo ""
    echo "Redeploying (delete + deploy)..."
    cmd_delete
    echo ""
    cmd_deploy
}

cmd_deploy() {
    echo "============================================================"
    echo "  GKE Multi-Dataset Deploy"
    echo "============================================================"
    echo "  Project:    ${PROJECT}"
    echo "  YAML:       ${JOB_YAML}"
    echo "  Image:      $(grep 'image:' "${JOB_YAML}" | awk '{print $2}')"
    echo "  Datasets:   ${DATASETS}"
    echo "  Clusters:   ${#CLUSTER_ARRAY[@]}"
    echo "  Timeout:    ${SCHEDULE_TIMEOUT}s per dataset"
    echo "============================================================"

    RESULTS=()

    for ds in "${DATASET_ARRAY[@]}"; do
        deploy_dataset "${JOB_YAML}" "${BASE_NAME}" "${ds}"
        RESULTS+=("${ds}")
    done

    echo ""
    echo "============================================================"
    echo "  All ${#RESULTS[@]} dataset(s) dispatched."
    echo "============================================================"
    echo ""
    echo "Check status:  bash gke/deploy.sh --status"
    echo "Delete all:    bash gke/deploy.sh --delete"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "${1:-}" in
    --status)
        cmd_status
        ;;
    --delete)
        cmd_delete
        ;;
    --redeploy)
        cmd_redeploy
        ;;
    --build)
        cmd_build_and_redeploy
        ;;
    --help|-h)
        cat <<'EOF'
Usage: bash gke/deploy.sh [COMMAND]

Commands:
  (none)       Deploy datasets to clusters (skip if already running)
  --build      Build image, push to AR, update YAML, delete old jobs, redeploy
  --redeploy   Delete existing jobs and redeploy with current YAML
  --status     Show job status across all clusters
  --delete     Delete all jobs from all clusters
  --help       Show this help

Environment variables:
  JOB_YAML          Path to job YAML (default: gke/regen-gemma4-26b.yaml)
  DATASETS          Comma-separated dataset names (default: magpie-multilingual,perfectblend)
  GPU_TYPES         Comma-separated GPU types to try (default: nvidia-h200-141gb,nvidia-h100-80gb)
                    Jobs are submitted for each type; first to schedule wins.
  CLUSTERS          Comma-separated cluster:region pairs (default: auto-discover)
  PROJECT           GCP project ID (default: cloud-llm-test)
  SCHEDULE_TIMEOUT  Seconds to wait per dataset for scheduling (default: 600)

Examples:
  bash gke/deploy.sh                                        # deploy, try H200 + H100
  GPU_TYPES=nvidia-h100-80gb bash gke/deploy.sh              # deploy on H100 only
  JOB_YAML=gke/regen-gemma3-27b.yaml bash gke/deploy.sh     # deploy gemma3
  JOB_YAML=gke/regen-gemma3-27b.yaml bash gke/deploy.sh --build  # build + deploy gemma3
EOF
        ;;
    *)
        cmd_deploy
        ;;
esac
