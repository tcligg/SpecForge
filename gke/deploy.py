#!/usr/bin/env python3
"""Operator-shortcut CLI for inspecting and tearing down regen jobs.

Step 8 of gke/PLAN.md reduced this script to a deprecation shim. The
end-to-end pipeline now lives in ``gke/orchestrator.py``; what stays
here are two ops shortcuts that are useful when the orchestrator is
not the right tool for the job:

  ``--status``   — show pod state across every (cluster x gpu) combo
                   for the datasets in the YAML. Cheap, side-effect
                   free, doesn't touch state.json.
  ``--delete``   — kubectl delete every job that matches the YAML's
                   datasets across every (cluster x gpu) combo. The
                   blunt-force "stop everything" hammer.

  ``--execute``  — DEPRECATED. Re-execs ``gke/orchestrator.py run``
                   so legacy callers don't break. Will be removed.

Pre-step-8 versions of this file ran the full pipeline themselves
(build, deploy_dataset, monitor, merge) with the candidate-racing
pattern. Steps 4a-7 of PLAN.md replaced that flow piece by piece;
step 8 is the cleanup.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gke.lib.clusters import use_cluster  # noqa: E402
from gke.lib.config import Config  # noqa: E402
from gke.lib.k8s import get_pod_status, run_output  # noqa: E402
from gke.lib.yaml_patch import gpu_short_name, make_job_name  # noqa: E402

# ---------------------------------------------------------------------------
# Ops shortcuts kept after step 8.
# ---------------------------------------------------------------------------


def cmd_status(cfg: Config) -> None:
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
                    ["kubectl", "get", "job", job_name, "--no-headers"]
                )
                comp_str = completions.split()[1] if completions else "not found"
                print(
                    f"    {job_name}: completions={comp_str}  "
                    f"pods: {running} running, {pending} pending, {other} other"
                )
    print()


def cmd_delete(cfg: Config) -> None:
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


# ---------------------------------------------------------------------------
# Deprecated --execute redirect.
# ---------------------------------------------------------------------------


def cmd_execute_redirect(args: argparse.Namespace) -> None:
    """Print a deprecation notice and re-exec the orchestrator.

    The orchestrator's flag surface differs from this CLI's, so we
    only forward the few flags that map cleanly: --yaml -> --config,
    --datasets, --output-dir, --project, --timeout. Anything else
    (including --gpu-types, --clusters) the orchestrator will discover
    or default on its own.
    """
    print()
    print("=" * 60)
    print("  gke/deploy.py --execute is DEPRECATED.")
    print("  Re-executing as: python3 gke/orchestrator.py run ...")
    print("  See gke/PLAN.md for the migration story.")
    print("=" * 60)
    print()

    orch = _PROJECT_ROOT / "gke" / "orchestrator.py"
    if not orch.is_file():
        sys.exit(f"Error: orchestrator not found at {orch}")

    cmd = [sys.executable, str(orch), "run", "--config", args.yaml]
    if args.datasets:
        cmd += ["--datasets", args.datasets]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    if args.project:
        cmd += ["--project", args.project]
    if args.gpu_types:
        cmd += ["--gpu-types", args.gpu_types]
    if args.clusters:
        cmd += ["--clusters", args.clusters]
    if args.timeout:
        cmd += ["--timeout", str(args.timeout)]

    print(f"$ {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operator-shortcut CLI for the SpecForge GKE regen "
        "pipeline. Use gke/orchestrator.py for the full flow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--yaml",
        required=True,
        help="Path to job YAML (e.g. gke/regen-gemma4-26b.yaml).",
    )
    parser.add_argument(
        "--datasets",
        default="magpie-multilingual,perfectblend",
        help="Comma-separated dataset names "
        "(default: magpie-multilingual,perfectblend).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="GCS output directory (gs://...).",
    )
    parser.add_argument(
        "--gpu-types",
        default="nvidia-h200-141gb,nvidia-h100-80gb",
        help="Comma-separated GPU types to inspect "
        "(default: nvidia-h200-141gb,nvidia-h100-80gb).",
    )
    parser.add_argument(
        "--clusters",
        default=None,
        help="Comma-separated cluster:region pairs (default: auto-discover).",
    )
    parser.add_argument(
        "--project",
        default="cloud-llm-test",
        help="GCP project ID (default: cloud-llm-test).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-dataset schedule timeout (default: 3600). Only used "
        "by the deprecated --execute redirect.",
    )

    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument(
        "--status",
        action="store_true",
        help="Show job status across all clusters.",
    )
    cmd_group.add_argument(
        "--delete",
        action="store_true",
        help="Delete all jobs from all clusters.",
    )
    cmd_group.add_argument(
        "--execute",
        action="store_true",
        help="DEPRECATED — re-execs `gke/orchestrator.py run`.",
    )

    args = parser.parse_args()

    if args.execute:
        cmd_execute_redirect(args)
        return  # unreachable: execvp replaces the process

    cfg = Config.from_args(args)

    if args.status:
        cmd_status(cfg)
    elif args.delete:
        cmd_delete(cfg)


if __name__ == "__main__":
    main()
