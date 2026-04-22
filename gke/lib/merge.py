"""Chunk merging via gsutil compose.

Extracted verbatim from gke/deploy.py in step 2. No logic changes.

Rename: _merge_output_chunks -> merge_output_chunks.
gke/deploy.py re-exports the old underscore name for compat.

Known limitation acknowledged in PLAN.md: silently truncates at
1024 chunks. Step 4b adds recursive compose to remove this limit.
"""

from __future__ import annotations

import re
import subprocess

from gke.lib.config import Config
from gke.lib.k8s import run_output


def merge_output_chunks(cfg: Config, dataset: str):
    """Merges output chunks in GCS using gsutil compose."""
    # This requires gsutil to be installed and authenticated.
    # Extract GCS bucket and base path from YAML's OUTPUT_DIR
    output_dir = cfg.output_dir
    if not output_dir:
        with open(cfg.job_yaml) as f:
            yaml_content = f.read()
        match = re.search(r'name: OUTPUT_DIR\s+value: "/gcs/([^"]+)"', yaml_content)
        if not match:
            print(
                f"Warning: Could not determine OUTPUT_DIR from "
                f"{cfg.job_yaml}. Cannot merge."
            )
            return
        output_dir = "gs://" + match.group(1)

    bucket_name = output_dir.replace("gs://", "").split("/")[0]
    base_path = "/".join(output_dir.replace("gs://", "").split("/")[1:])

    # The input file stem is used as the dataset name in regen_entrypoint.py
    # We need to find what the original file was.
    # This is a simplification; a more robust solution might need to check
    # INPUT_FILES or PREPARE_DATA_DIR. For now, assume common names.
    if dataset == "magpie-multilingual":
        dataset_name_stem = "translate-bp-dataset_train"
    else:
        dataset_name_stem = f"{dataset}_train"

    chunks_dir = f"gs://{bucket_name}/{base_path}/{dataset_name_stem}"
    merged_file = f"gs://{bucket_name}/{base_path}/{dataset}_regen.jsonl"

    print(f"  Composing chunks from {chunks_dir} into {merged_file}...")

    # Get list of chunks
    chunk_list_str = run_output(["gsutil", "ls", f"{chunks_dir}/chunk_*.jsonl"])
    if not chunk_list_str:
        print(f"Warning: No chunk files found at {chunks_dir}. Nothing to merge.")
        return

    chunks = chunk_list_str.splitlines()
    if len(chunks) > 1024:
        print(
            f"Warning: >1024 chunks found for {dataset}. "
            f"gsutil compose has a limit of 1024."
        )
        print("  Merging first 1024 chunks only.")
        chunks = chunks[:1024]

    # Compose command
    compose_cmd = ["gsutil", "-m", "compose"] + chunks + [merged_file]

    # Delete the merged file if it exists, as compose fails otherwise
    run_output(["gsutil", "rm", merged_file])

    result = subprocess.run(compose_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: gsutil compose failed for {dataset}.")
        print(result.stderr)
    else:
        print(f"  Successfully merged {len(chunks)} chunks for {dataset}.")
        # Optional: delete the source chunks
        # run_output(["gsutil", "-m", "rm"] + chunks)
