"""YAML patching for per-dataset job submissions.

Extracted verbatim from gke/deploy.py in step 2. No logic changes.

The string-based patcher is acknowledged as fragile in PLAN.md;
replacing with ruamel.yaml is a deferred item.
"""

from __future__ import annotations

import re
import subprocess
from typing import Tuple

from gke.lib.config import Config


def gpu_short_name(gpu_type: str) -> str:
    """nvidia-h200-141gb -> h200"""
    return gpu_type.replace("nvidia-", "").split("-")[0]


def make_job_name(base: str, dataset: str, gpu_type: str, region: str) -> str:
    """e.g. regen-gemma4-26b-perfectblend-h200-europe-west1"""
    return f"{base}-{dataset}-{gpu_short_name(gpu_type)}-{region}"


def patch_and_apply_yaml(
    cfg: Config,
    dataset: str,
    gpu_type: str,
    region: str,
) -> Tuple[bool, str]:
    """Patch YAML and apply via kubectl. Returns (success, error_message)."""
    job_name = make_job_name(cfg.base_name, dataset, gpu_type, region)

    with open(cfg.job_yaml) as f:
        content = f.read()

    # Patch job name (first occurrence)
    content = content.replace(f"name: {cfg.base_name}", f"name: {job_name}", 1)

    # Patch DATASETS value and GPU type
    lines = content.splitlines()
    patched_lines = []
    in_datasets_env = False
    in_output_dir_env = False

    for line in lines:
        if in_datasets_env and "value:" in line:
            line = re.sub(r'value: ".*"', f'value: "{dataset}"', line)
            in_datasets_env = False
        elif in_output_dir_env and "value:" in line and cfg.output_dir:
            # Convert gs://bucket/path -> /gcs/bucket/path
            gcs_path = cfg.output_dir.replace("gs://", "/gcs/")
            line = re.sub(r'value: ".*"', f'value: "{gcs_path}"', line)
            in_output_dir_env = False

        if "name: DATASETS" in line:
            in_datasets_env = True
        elif "name: OUTPUT_DIR" in line:
            in_output_dir_env = True

        if "gke-accelerator:" in line and "accelerator-count" not in line:
            line = re.sub(r"gke-accelerator: .*", f"gke-accelerator: {gpu_type}", line)

        patched_lines.append(line)

    patched_content = "\n".join(patched_lines) + "\n"

    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=patched_content,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, ""
