#!/usr/bin/env bash
# Entrypoint for SpecForge data regeneration on GKE batch jobs.
#
# This script:
#   1. Launches SGLang server(s) for the target model.
#   2. Waits for all servers to become healthy.
#   3. Runs regenerate_train_data.py with chunk-based processing.
#      Sharding across pods is handled by the Python script via
#      --shard-id / --total-shards (derived from JOB_COMPLETION_INDEX).
#   4. Cleans up servers on exit.
#
# Required environment variables:
#   JOB_COMPLETION_INDEX  - Set by GKE indexed Job (0-based shard index)
#   TOTAL_SHARDS          - Total number of job completions
#   MODEL                 - HuggingFace model ID (e.g. google/gemma-4-26b-a4b-it)
#   INPUT_FILE            - Path to input JSONL file
#   OUTPUT_DIR            - Directory to write chunk output files
#
# Optional environment variables (with defaults):
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

# ── Validate required env vars ───────────────────────────────────────────────
: "${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX is required}"
: "${TOTAL_SHARDS:?TOTAL_SHARDS is required}"
: "${MODEL:?MODEL is required}"
: "${INPUT_FILE:?INPUT_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

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
echo "  Input file:        ${INPUT_FILE}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Mem fraction:      ${MEM_FRACTION_STATIC}"
echo "============================================================"

# ── Launch SGLang servers ────────────────────────────────────────────────────
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

# ── Wait for servers ─────────────────────────────────────────────────────────
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

# ── Run regeneration ─────────────────────────────────────────────────────────
echo "Starting chunk-based data regeneration..."

REGEN_CMD=(
    python3 /app/specforge/scripts/regenerate_train_data.py
    --model "${MODEL}"
    --concurrency "${CONCURRENCY}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --server-address "${SERVER_ADDRESSES[@]}"
    --input-file-path "${INPUT_FILE}"
    --output-dir "${OUTPUT_DIR}"
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
echo "============================================================"
echo "  Shard ${JOB_COMPLETION_INDEX} complete!"
echo "============================================================"
