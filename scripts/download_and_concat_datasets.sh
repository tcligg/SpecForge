#!/bin/bash
# Download and preformat chunked datasets from GCS (via gcsfuse).
# Usage: bash scripts/download_and_concat_datasets.sh

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
OUTPUT_DIR="$ROOT_DIR/outputs/dataset"

GCS_BASE="$HOME/gcs/pyc-eagle-data/datasets/gemma4-26b-regen"

# source_dir:output_file
DATASETS=(
    "perfectblend_train:perfectblend_regen_gemma4_preformatted.jsonl"
    "translate-bp-dataset_train:translate_bp_regen_gemma4_preformatted.jsonl"
)

mkdir -p "$OUTPUT_DIR"

for entry in "${DATASETS[@]}"; do
    IFS=':' read -r src_dir dst_file <<< "$entry"
    dst_path="$OUTPUT_DIR/$dst_file"

    if [[ -f "$dst_path" ]]; then
        echo "[$dst_file] Already exists ($(wc -l < "$dst_path") lines). Skipping."
        continue
    fi

    echo "[$dst_file] Converting from $src_dir ..."
    python3 "$ROOT_DIR/scripts/convert_sharegpt_to_preformatted.py" \
        --input "$GCS_BASE/$src_dir" \
        --output "$dst_path" \
        --format gemma4
done

echo ""
echo "All datasets ready in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*preformatted*.jsonl
