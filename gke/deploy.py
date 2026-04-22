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
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gke import state as _state
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

    # Pull the first non-comment ``image:`` key. After step 4b the
    # YAMLs carry an explanatory comment that includes ``image:``;
    # match against a leading-whitespace line so we skip it.
    image = "unknown"
    with open(cfg.job_yaml) as f:
        for line in f:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "image:" in line and not stripped.startswith("- "):
                image = line.strip().split("image:", 1)[-1].strip()
                break

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


# ---------------------------------------------------------------------------
# State observer (step 3 of gke/PLAN.md)
#
# When --state-uri is set, cmd_execute writes a per-phase observer
# record to the GCS path. This is *pure observation*: failures to write
# state never break the workflow, and nothing reads state in step 3.
# Step 4 starts using the read path.
# ---------------------------------------------------------------------------


def _make_state_store(cfg: Config) -> Optional["_state.StateStore"]:
    """Construct a StateStore if --state-uri was provided, else None.

    Construction errors (bad URI) are logged and swallowed: state
    collection is opt-in and must not block the workflow.
    """
    if not cfg.state_uri:
        return None
    try:
        return _state.StateStore(cfg.state_uri)
    except _state.StateError as exc:
        print(f"  Warning: --state-uri rejected, state collection disabled: {exc}")
        return None


def _observe(
    store: Optional["_state.StateStore"],
    fn,
    *,
    label: str,
) -> None:
    """Apply a state mutation through the store, swallowing errors.

    Pure-observer contract: a state-write failure prints a warning and
    returns; it must never propagate up into the workflow loop.
    """
    if store is None:
        return
    try:
        store.update(fn)
    except _state.StateError as exc:
        print(f"  Warning: state {label}: {exc}")


def cmd_execute(cfg: Config):
    """End-to-end workflow: build, push, deploy, wait, merge.

    When --state-uri is set, writes an observer record to the configured
    GCS path at each phase boundary (build/deploy/monitor/merge). State
    is write-only in step 3; the orchestrator added in step 4 is the
    first reader.
    """
    print()
    print("============================================================")
    print("  Executing End-to-End Regeneration Workflow")
    print("============================================================")

    # 0. Initialize state observer (optional, gated on --state-uri).
    store = _make_state_store(cfg)
    if store is not None:
        try:
            run_id = _state.compute_run_id(cfg.base_name or "regen")
            initial = _state.initial_state(
                run_id=run_id,
                config_yaml=cfg.job_yaml,
                config_hash=_state.compute_config_hash(cfg.job_yaml),
            )
            store.write(initial)
            print(f"  State: writing observer records to {cfg.state_uri}")
            print(f"  Run ID: {run_id}")
        except _state.StateError as exc:
            print(f"  Warning: state init failed; continuing without it: {exc}")
            store = None

    # 1. Build and push image, update YAML
    new_image_uri = build_and_push_image(cfg)
    with open(cfg.job_yaml) as f:
        content = f.read()
    # Match the first non-comment ``image:`` key. Step 4b unpinned the
    # YAMLs and added an explanatory comment that contains the literal
    # ``image:`` substring; without the line anchor + leading-whitespace
    # constraint we'd rewrite the comment instead.
    content = re.sub(
        r"(?m)^(?P<lead>[ \t]+image:[ \t]*)\S+",
        rf"\g<lead>{new_image_uri}",
        content,
        count=1,
    )
    with open(cfg.job_yaml, "w") as f:
        f.write(content)
    print(f"Updated {cfg.job_yaml} to use image {new_image_uri}")

    def _record_build(s: "_state.RunState") -> "_state.RunState":
        s.phases.build = _state.BuildPhase(
            image_uri=new_image_uri,
            git_sha="",  # populated in step 4b when Phase B is host-side
            dirty=False,
            completed_at=_state._utc_now_iso(),
        )
        return s

    _observe(store, _record_build, label="build")

    # 2. Delete any old jobs
    cmd_delete(cfg)

    # 3. Deploy, wait, and merge for each dataset
    for ds in cfg.datasets:
        winner_job = deploy_dataset(cfg, ds)
        if not winner_job:
            print(f"  Warning: Failed to deploy dataset {ds}. Skipping.")
            continue

        job_name, cluster, region = winner_job

        def _record_deploy(
            s: "_state.RunState",
            *,
            ds=ds,
            job_name=job_name,
            cluster=cluster,
            region=region,
        ) -> "_state.RunState":
            s.phases.deploy[ds] = {
                "primary_job": {
                    "cluster": cluster,
                    "region": region,
                    # GPU type is not surfaced by deploy_dataset's return
                    # value; recover it from the job_name suffix.
                    "gpu": "",
                    "job_name": job_name,
                    "total_shards": cfg.total_pods,
                    "started_at": _state._utc_now_iso(),
                }
            }
            return s

        _observe(store, _record_deploy, label=f"deploy[{ds}]")

        print(f"  Monitoring job {job_name} in {cluster}...")
        success = wait_for_job_completion(job_name, cluster, region, cfg)

        def _record_monitor(
            s: "_state.RunState",
            *,
            ds=ds,
            success=success,
        ) -> "_state.RunState":
            s.phases.monitor[ds] = {
                "primary_job_status": "Complete" if success else "Failed",
                # Per-shard status arrives in step 6 once we read
                # .status.completedIndexes; record empty for now.
                "shard_status": {},
            }
            return s

        _observe(store, _record_monitor, label=f"monitor[{ds}]")

        if not success:
            print(f"  Error: Job {job_name} failed or timed out. Skipping merge.")
            continue

        print(f"  Job {job_name} completed. Merging output chunks...")
        merge_output_chunks(cfg, ds)

        def _record_merge(
            s: "_state.RunState",
            *,
            ds=ds,
        ) -> "_state.RunState":
            # The current merge_output_chunks doesn't return chunk count
            # or merged-file URI; record a minimal completion marker.
            # Step 4b's Phase E rewrite will populate the full schema.
            s.phases.merge[ds] = {
                "completed_at": _state._utc_now_iso(),
            }
            return s

        _observe(store, _record_merge, label=f"merge[{ds}]")

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
    parser.add_argument(
        "--state-uri",
        default=None,
        help="Optional gs://bucket/path/state.json to write per-phase "
        "observer records. Step 3 of gke/PLAN.md: write-only; the "
        "orchestrator added in step 4 starts using the read path.",
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
