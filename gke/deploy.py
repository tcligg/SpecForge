#!/usr/bin/env python3
"""
Deploy per-dataset regen batch jobs across multiple GKE clusters.
For each dataset, the script:
  1. Patches the YAML with that single dataset, GPU type, and a unique job name.
  2. Submits to ALL (cluster x gpu_type) combinations.
  3. Waits until one combination has enough pods Running.
  4. Deletes the job from all other combinations.
  5. Moves on to the next dataset.

Usage:
  python gke/deploy.py --yaml gke/regen-gemma4-26b.yaml
  python gke/deploy.py --yaml gke/regen-gemma4-26b.yaml --status
  python gke/deploy.py --yaml gke/regen-gemma4-26b.yaml --delete
  python gke/deploy.py --yaml gke/regen-gemma4-26b.yaml --execute
  python gke/deploy.py --yaml gke/regen-gemma4-26b.yaml --gpu-types nvidia-h100-80gb

NOTE (step 2 of gke/PLAN.md): the implementation has been split into
gke/lib/* modules. This file is now a thin CLI shim that wires argparse
to the library helpers. Step 6 will replace this CLI entirely with
gke/orchestrator.py.
"""

from __future__ import annotations

import argparse
import re

# Import paths: deploy.py runs as a top-level script, so make sure the
# project root is on sys.path before importing the gke.lib package.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gke.lib.build import build_and_push_image
from gke.lib.clusters import use_cluster
from gke.lib.config import Config
from gke.lib.k8s import get_pod_status, run_output, wait_for_job_completion
from gke.lib.merge import merge_output_chunks
from gke.lib.orchestration import deploy_dataset
from gke.lib.yaml_patch import gpu_short_name, make_job_name

# Backward-compatible aliases for the underscore-prefixed names that
# step 2 dropped. Keeping these so any external caller that did
# `from gke.deploy import _build_and_push_image` keeps working until
# step 8 deletes the CLI entirely.
_build_and_push_image = build_and_push_image
_merge_output_chunks = merge_output_chunks
_wait_for_job_completion = wait_for_job_completion


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_status(cfg: Config):
    print()
    print("Status for all dataset jobs:")
    print("-" * 60)
    for ds in cfg.datasets:
        print(f"  Dataset: {ds}")
        for gpu in cfg.gpu_types:
            for cluster, region in cfg.clusters:
                job_name = make_job_name(cfg.base_name, ds, gpu, region)
                if not use_cluster(cluster, region, cfg.project):
                    continue
                running, pending, other = get_pod_status(job_name)
                if running + pending + other == 0:
                    continue
                completions = run_output(
                    [
                        "kubectl",
                        "get",
                        "job",
                        job_name,
                        "--no-headers",
                    ]
                )
                comp_str = completions.split()[1] if completions else "not found"
                print(
                    f"    {job_name}: completions={comp_str}  "
                    f"pods: {running} running, {pending} pending, {other} other"
                )
    print()


def cmd_delete(cfg: Config):
    print("Deleting all dataset jobs from all clusters...")
    for ds in cfg.datasets:
        for gpu in cfg.gpu_types:
            for cluster, region in cfg.clusters:
                job_name = make_job_name(cfg.base_name, ds, gpu, region)
                if use_cluster(cluster, region, cfg.project):
                    result = run_output(
                        [
                            "kubectl",
                            "delete",
                            "job",
                            job_name,
                            "--ignore-not-found",
                        ]
                    )
                    if result and "deleted" in result:
                        print(f"  {job_name}: deleted.")
    print("Done.")


def cmd_deploy(cfg: Config):
    print("=" * 60)
    print("  GKE Multi-Dataset Deploy")
    print("=" * 60)

    with open(cfg.job_yaml) as f:
        for line in f:
            if "image:" in line:
                image = line.strip().split("image:")[-1].strip()
                break
        else:
            image = "unknown"

    print(f"  Project:    {cfg.project}")
    print(f"  YAML:       {cfg.job_yaml}")
    print(f"  Image:      {image}")
    print(f"  Datasets:   {', '.join(cfg.datasets)}")
    print(f"  GPU types:  {', '.join(gpu_short_name(g) for g in cfg.gpu_types)}")
    print(f"  Clusters:   {len(cfg.clusters)}")
    print(f"  Timeout:    {cfg.schedule_timeout}s per dataset")
    if cfg.output_dir:
        print(f"  Output Dir: {cfg.output_dir}")
    print("=" * 60)

    for ds in cfg.datasets:
        deploy_dataset(cfg, ds)

    print()
    print("=" * 60)
    print(f"  All {len(cfg.datasets)} dataset(s) dispatched.")
    print("=" * 60)


def cmd_execute(cfg: Config):
    """End-to-end workflow: build, push, deploy, wait, merge."""
    print()
    print("============================================================")
    print("  Executing End-to-End Regeneration Workflow")
    print("============================================================")

    # 1. Build and push image, update YAML
    new_image_uri = build_and_push_image(cfg)
    with open(cfg.job_yaml) as f:
        content = f.read()
    content = re.sub(r"(image:\s*)(\S+)", rf"\g<1>{new_image_uri}", content, count=1)
    with open(cfg.job_yaml, "w") as f:
        f.write(content)
    print(f"Updated {cfg.job_yaml} to use image {new_image_uri}")

    # 2. Delete any old jobs
    cmd_delete(cfg)

    # 3. Deploy, wait, and merge for each dataset
    for ds in cfg.datasets:
        winner_job = deploy_dataset(cfg, ds)
        if not winner_job:
            print(f"  Warning: Failed to deploy dataset {ds}. Skipping.")
            continue

        job_name, cluster, region = winner_job
        print(f"  Monitoring job {job_name} in {cluster}...")
        success = wait_for_job_completion(job_name, cluster, region, cfg)
        if not success:
            print(f"  Error: Job {job_name} failed or timed out. Skipping merge.")
            continue

        print(f"  Job {job_name} completed. Merging output chunks...")
        merge_output_chunks(cfg, ds)

    print()
    print("============================================================")
    print("  Workflow Complete")
    print("============================================================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Deploy per-dataset regen batch jobs across GKE clusters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config flags
    parser.add_argument(
        "--yaml",
        required=True,
        help="Path to job YAML (e.g. gke/regen-gemma4-26b.yaml)",
    )
    parser.add_argument(
        "--datasets",
        default="magpie-multilingual,perfectblend",
        help="Comma-separated dataset names "
        "(default: magpie-multilingual,perfectblend)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="GCS output directory (e.g. gs://my-bucket/output). Overrides YAML.",
    )
    parser.add_argument(
        "--gpu-types",
        default="nvidia-h200-141gb,nvidia-h100-80gb",
        help="Comma-separated GPU types to try "
        "(default: nvidia-h200-141gb,nvidia-h100-80gb)",
    )
    parser.add_argument(
        "--clusters",
        default=None,
        help="Comma-separated cluster:region pairs (default: auto-discover)",
    )
    parser.add_argument(
        "--project",
        default="cloud-llm-test",
        help="GCP project ID (default: cloud-llm-test)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds to wait per dataset for scheduling (default: 3600)",
    )

    # Command flags (mutually exclusive)
    cmd_group = parser.add_mutually_exclusive_group()
    cmd_group.add_argument(
        "--status", action="store_true", help="Show job status across all clusters"
    )
    cmd_group.add_argument(
        "--delete", action="store_true", help="Delete all jobs from all clusters"
    )
    cmd_group.add_argument(
        "--execute",
        action="store_true",
        help="Run the full workflow: build, push, deploy, wait, and merge",
    )

    args = parser.parse_args()
    cfg = Config.from_args(args)

    if args.status:
        cmd_status(cfg)
    elif args.delete:
        cmd_delete(cfg)
    elif args.execute:
        cmd_execute(cfg)
    else:
        cmd_deploy(cfg)


if __name__ == "__main__":
    main()
