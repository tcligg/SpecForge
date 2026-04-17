#!/usr/bin/env bash
# Entrypoint for SpecForge data regeneration on GKE batch jobs.
#
# This script:
#   1. Prepares datasets (downloads from HuggingFace, converts to JSONL).
#   2. Launches SGLang server(s) for the target model.
#   3. Waits for all servers to become healthy.
#   4. Runs regenerate_train_data.py for each dataset with chunk-based
#      processing. Sharding across pods is handled by the Python script
#      via --shard-id / --total-shards.
#   5. Cleans up servers on exit.
#
# Required environment variables:
#   JOB_COMPLETION_INDEX  - Set by GKE indexed Job (0-based shard index)
#   TOTAL_SHARDS          - Total number of job completions
#   MODEL                 - HuggingFace model ID (e.g. google/gemma-4-26b-a4b-it)
#   DATASETS              - Comma-separated dataset names for prepare_data.py
#                           e.g. ultrachat,perfectblend
#                           Choices: ultrachat, sharegpt, eaglechat, perfectblend,
#                           opc, gsm8k, hendrycks_math, math_qa, codealpaca-20k,
#                           opencodeinstruct, magicoder-evol-instruct, sciq, camel, etc.
#   OUTPUT_DIR            - Base directory for output (per-dataset subdirs created)
#
# Optional environment variables (with defaults):
#   PREPARE_DATA_DIR      - Where prepare_data.py writes JONLs (default: OUTPUT_DIR/prepared)
#   INPUT_FILES           - If set, skip prepare_data.py and use these comma-separated
#                           JSONL paths directly (overrides DATASETS)
#   TP_SIZE               - Tensor parallel size (default: auto from GPU count)
#   NUM_SERVERS           - Number of SGLang server instances (default: 1)
#   BASE_PORT             - First server port (default: 30000)
#   CONCURRENCY           - Requests per server (default: 128)
#   MAX_TOKENS            - Max generation tokens (default: 2048)
#   TEMPERATURE           - Sampling temperature (default: 1.0)
#   IS_REASONING_MODEL    - Set to "true" for reasoning models (default: true)
#   THINKING_RATIO        - Fraction of requests with thinking (default: 0.7)
#   CHUNK_SIZE            - Samples per chunk (default: 500)
#   MEM_FRACTION_STATIC   - SGLang memory fraction (default: 0.85)
#   CONTEXT_LENGTH        - SGLang context length (default: unset)
#   EXTRA_SGLANG_ARGS     - Additional args for sglang.launch_server
#   EXTRA_REGEN_ARGS      - Additional args for regenerate_train_data.py

set -euo pipefail

SPECFORGE_DIR="/app/specforge"

# ── Validate required env vars ───────────────────────────────────────────────
: "${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX is required}"
: "${TOTAL_SHARDS:?TOTAL_SHARDS is required}"
: "${MODEL:?MODEL is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

if [ -z "${DATASETS:-}" ] && [ -z "${INPUT_FILES:-}" ]; then
    echo "Error: Either DATASETS or INPUT_FILES must be set."
    exit 1
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
AVAIL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
NUM_SERVERS="${NUM_SERVERS:-1}"
TP_SIZE="${TP_SIZE:-$(( AVAIL_GPUS / NUM_SERVERS ))}"
BASE_PORT="${BASE_PORT:-30000}"
CONCURRENCY="${CONCURRENCY:-128}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
IS_REASONING_MODEL="${IS_REASONING_MODEL:-true}"
THINKING_RATIO="${THINKING_RATIO:-0.7}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
PREPARE_DATA_DIR="${PREPARE_DATA_DIR:-${OUTPUT_DIR}/prepared}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"
EXTRA_REGEN_ARGS="${EXTRA_REGEN_ARGS:-}"

echo "============================================================"
echo "  SpecForge Data Regeneration"
echo "  Shard ${JOB_COMPLETION_INDEX} / ${TOTAL_SHARDS}"
echo "============================================================"
echo "  Model:             ${MODEL}"
echo "  TP size:           ${TP_SIZE}"
echo "  Num servers:       ${NUM_SERVERS}"
echo "  Available GPUs:    ${AVAIL_GPUS}"
echo "  Concurrency:       ${CONCURRENCY}"
echo "  Max tokens:        ${MAX_TOKENS}"
echo "  Temperature:       ${TEMPERATURE}"
echo "  Reasoning model:   ${IS_REASONING_MODEL}"
echo "  Thinking ratio:    ${THINKING_RATIO}"
echo "  Chunk size:        ${CHUNK_SIZE}"
echo "  Datasets:          ${DATASETS:-<using INPUT_FILES>}"
echo "  Input files:       ${INPUT_FILES:-<from prepare_data>}"
echo "  Prepare data dir:  ${PREPARE_DATA_DIR}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Mem fraction:      ${MEM_FRACTION_STATIC}"
echo "============================================================"

# ── Step 1: Prepare datasets ────────────────────────────────────────────────
# Only the first shard prepares data; others wait for the files to appear.
# prepare_data.py is idempotent (skips if output already exists).

if [ -n "${DATASETS:-}" ] && [ -z "${INPUT_FILES:-}" ]; then
    IFS=',' read -ra DATASET_ARRAY <<< "${DATASETS}"

    echo ""
    echo "Preparing ${#DATASET_ARRAY[@]} dataset(s)..."
    mkdir -p "${PREPARE_DATA_DIR}"

    for ds_name in "${DATASET_ARRAY[@]}"; do
        OUTPUT_JSONL="${PREPARE_DATA_DIR}/${ds_name}_train.jsonl"

        if [ -f "${OUTPUT_JSONL}" ]; then
            LINES=$(wc -l < "${OUTPUT_JSONL}")
            echo "  ${ds_name}: already prepared (${LINES} lines), skipping."
            continue
        fi

        # Only shard 0 runs prepare_data to avoid redundant downloads
        if [ "${JOB_COMPLETION_INDEX}" -eq 0 ]; then
            echo "  ${ds_name}: running prepare_data.py..."
            python3 "${SPECFORGE_DIR}/scripts/prepare_data.py" \
                --dataset "${ds_name}" \
                --output-path "${PREPARE_DATA_DIR}"
        else
            echo "  ${ds_name}: waiting for shard 0 to prepare data..."
            WAIT_ELAPSED=0
            WAIT_MAX=1800  # 30 minutes
            while [ ! -f "${OUTPUT_JSONL}" ] && [ "${WAIT_ELAPSED}" -lt "${WAIT_MAX}" ]; do
                sleep 10
                WAIT_ELAPSED=$(( WAIT_ELAPSED + 10 ))
            done
            if [ ! -f "${OUTPUT_JSONL}" ]; then
                echo "Error: ${OUTPUT_JSONL} not found after ${WAIT_MAX}s."
                exit 1
            fi
        fi

        echo "  ${ds_name}: ready."
    done

    # Build INPUT_FILES from prepared data
    INPUT_FILES_BUILT=""
    for ds_name in "${DATASET_ARRAY[@]}"; do
        JSONL="${PREPARE_DATA_DIR}/${ds_name}_train.jsonl"
        if [ -n "${INPUT_FILES_BUILT}" ]; then
            INPUT_FILES_BUILT="${INPUT_FILES_BUILT},${JSONL}"
        else
            INPUT_FILES_BUILT="${JSONL}"
        fi
    done
    INPUT_FILES="${INPUT_FILES_BUILT}"

    echo ""
    echo "All datasets prepared. Input files: ${INPUT_FILES}"
    echo "------------------------------------------------------------"
fi

# ── Step 2: Launch SGLang servers ────────────────────────────────────────────
SERVER_PIDS=()
SERVER_ADDRESSES=()

cleanup() {
    echo ""
    echo "Shutting down SGLang server(s)..."
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 3
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "All servers stopped."
}
trap cleanup EXIT

for i in $(seq 0 $(( NUM_SERVERS - 1 ))); do
    PORT=$(( BASE_PORT + i * 10 ))
    GPU_START=$(( i * TP_SIZE ))
    GPU_END=$(( GPU_START + TP_SIZE - 1 ))
    CUDA_DEVICES=$(seq -s, "$GPU_START" "$GPU_END")

    echo "Starting server $((i+1))/${NUM_SERVERS} on GPUs ${CUDA_DEVICES}, port ${PORT}..."

    SGLANG_CMD=(
        python3 -m sglang.launch_server
        --model "${MODEL}"
        --tp "${TP_SIZE}"
        --port "${PORT}"
        --host 0.0.0.0
        --dtype bfloat16
        --cuda-graph-max-bs 128
        --trust-remote-code
        --mem-fraction-static "${MEM_FRACTION_STATIC}"
    )

    if [ -n "${CONTEXT_LENGTH}" ]; then
        SGLANG_CMD+=(--context-length "${CONTEXT_LENGTH}")
    fi

    if [ -n "${EXTRA_SGLANG_ARGS}" ]; then
        # shellcheck disable=SC2206
        SGLANG_CMD+=(${EXTRA_SGLANG_ARGS})
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${SGLANG_CMD[@]}" \
        > "/tmp/sglang_server_${PORT}.log" 2>&1 &

    SERVER_PIDS+=($!)
    SERVER_ADDRESSES+=("localhost:${PORT}")
done

# ── Step 3: Wait for servers ─────────────────────────────────────────────────
echo ""
echo "Waiting for servers to become healthy..."

wait_for_server() {
    local addr=$1
    local max_wait=900  # 15 minutes
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://${addr}/health" > /dev/null 2>&1; then
            return 0
        fi
        for pid in "${SERVER_PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Error: Server process $pid died."
                tail -50 /tmp/sglang_server_*.log
                return 1
            fi
        done
        sleep 5
        elapsed=$(( elapsed + 5 ))
    done
    echo "Error: Server ${addr} timed out after ${max_wait}s."
    tail -50 /tmp/sglang_server_*.log
    return 1
}

for addr in "${SERVER_ADDRESSES[@]}"; do
    if wait_for_server "$addr"; then
        echo "  ${addr} is healthy."
    else
        exit 1
    fi
done

echo "All ${NUM_SERVERS} server(s) are ready."
echo "------------------------------------------------------------"

# ── Step 4: Run regeneration for each dataset ────────────────────────────────
IFS=',' read -ra INPUT_FILE_ARRAY <<< "${INPUT_FILES}"
TOTAL_DATASETS=${#INPUT_FILE_ARRAY[@]}

echo "Starting chunk-based data regeneration for ${TOTAL_DATASETS} dataset(s)..."
echo ""

DATASET_INDEX=0
for INPUT_FILE in "${INPUT_FILE_ARRAY[@]}"; do
    DATASET_INDEX=$(( DATASET_INDEX + 1 ))

    # Derive dataset name from filename (without extension)
    DATASET_NAME=$(basename "${INPUT_FILE}" .jsonl)
    DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}"

    echo "============================================================"
    echo "  Dataset ${DATASET_INDEX}/${TOTAL_DATASETS}: ${DATASET_NAME}"
    echo "  Input:  ${INPUT_FILE}"
    echo "  Output: ${DATASET_OUTPUT_DIR}"
    echo "============================================================"

    if [ ! -f "${INPUT_FILE}" ]; then
        echo "Warning: ${INPUT_FILE} not found, skipping."
        continue
    fi

    REGEN_CMD=(
        python3 "${SPECFORGE_DIR}/scripts/regenerate_train_data.py"
        --model "${MODEL}"
        --concurrency "${CONCURRENCY}"
        --max-tokens "${MAX_TOKENS}"
        --temperature "${TEMPERATURE}"
        --server-address "${SERVER_ADDRESSES[@]}"
        --input-file-path "${INPUT_FILE}"
        --output-dir "${DATASET_OUTPUT_DIR}"
        --chunk-size "${CHUNK_SIZE}"
        --shard-id "${JOB_COMPLETION_INDEX}"
        --total-shards "${TOTAL_SHARDS}"
    )

    if [ "${IS_REASONING_MODEL}" = "true" ]; then
        REGEN_CMD+=(--is-reasoning-model)
        if [ -n "${THINKING_RATIO}" ]; then
            REGEN_CMD+=(--thinking-ratio "${THINKING_RATIO}")
        fi
    fi

    if [ -n "${EXTRA_REGEN_ARGS}" ]; then
        # shellcheck disable=SC2206
        REGEN_CMD+=(${EXTRA_REGEN_ARGS})
    fi

    echo "Running: ${REGEN_CMD[*]}"
    "${REGEN_CMD[@]}"

    echo ""
    echo "  Dataset ${DATASET_NAME} complete."
    echo ""
done

echo "============================================================"
echo "  Shard ${JOB_COMPLETION_INDEX} complete!"
echo "  All ${TOTAL_DATASETS} dataset(s) processed."
echo "============================================================"
