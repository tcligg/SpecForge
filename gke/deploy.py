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
  python gke/deploy.py --yaml gke/regen-gemma4-26b.yaml --build
  python gke/deploy.py --yaml gke/regen-gemma4-26b.yaml --gpu-types nvidia-h100-80gb
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


SCRIPT_DIR = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    project: str = "cloud-llm-test"
    schedule_timeout: int = 3600
    job_yaml: str = ""
    datasets: List[str] = field(default_factory=list)
    gpu_types: List[str] = field(default_factory=list)
    clusters: List[Tuple[str, str]] = field(default_factory=list)
    base_name: str = ""
    total_pods: int = 8
    min_running: int = 4

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        cfg = cls()
        cfg.project = args.project
        cfg.schedule_timeout = args.timeout
        cfg.job_yaml = args.yaml
        cfg.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        cfg.gpu_types = [g.strip() for g in args.gpu_types.split(",") if g.strip()]

        # Validate JOB_YAML
        if not cfg.job_yaml:
            available = sorted(glob.glob(str(SCRIPT_DIR / "regen-*.yaml")))
            print("Error: --yaml is required. Available YAMLs:")
            for f in available:
                print(f"  {f}")
            sys.exit(1)

        if not os.path.isfile(cfg.job_yaml):
            print(f"Error: {cfg.job_yaml} not found.")
            sys.exit(1)

        # Parse YAML for base name and completions
        with open(cfg.job_yaml) as f:
            yaml_content = f.read()

        match = re.search(r"^\s*name:\s*(\S+)", yaml_content, re.MULTILINE)
        cfg.base_name = match.group(1) if match else "regen"

        match = re.search(r"completions:\s*(\d+)", yaml_content)
        cfg.total_pods = int(match.group(1)) if match else 8
        cfg.min_running = (cfg.total_pods + 1) // 2

        # Discover or parse clusters
        if args.clusters:
            cfg.clusters = []
            for entry in args.clusters.split(","):
                entry = entry.strip()
                if ":" in entry:
                    name, region = entry.split(":", 1)
                    cfg.clusters.append((name.strip(), region.strip()))
        else:
            cfg.clusters = discover_clusters(cfg.project)

        return cfg


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def run_output(cmd: List[str]) -> str:
    """Run a command and return stdout, empty string on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def discover_clusters(project: str) -> List[Tuple[str, str]]:
    """Auto-discover pyc-batch* clusters, excluding pyc-batch-us-south1."""
    print(f"Discovering pyc-batch* clusters in project {project}...")
    output = run_output(
        [
            "gcloud",
            "container",
            "clusters",
            "list",
            "--project",
            project,
            "--filter",
            "name~^pyc-batch AND NOT name=pyc-batch-us-south1",
            "--format",
            "csv[no-heading](name,location)",
        ]
    )
    if not output:
        print(f"Error: No pyc-batch* clusters found in project {project}.")
        sys.exit(1)

    clusters = []
    for line in output.splitlines():
        parts = line.strip().split(",")
        if len(parts) == 2:
            clusters.append((parts[0].strip(), parts[1].strip()))
    return clusters


def use_cluster(cluster: str, region: str, project: str) -> bool:
    """Switch kubectl context to a cluster."""
    result = subprocess.run(
        [
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            cluster,
            "--region",
            region,
            "--project",
            project,
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def get_pod_status(job_name: str) -> Tuple[int, int, int]:
    """Get (running, pending, other) pod counts for a job."""
    output = run_output(
        [
            "kubectl",
            "get",
            "pods",
            "-l",
            f"job-name={job_name}",
            "--no-headers",
        ]
    )
    running = pending = other = 0
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            status = parts[2]
            if status == "Running":
                running += 1
            elif status == "Pending":
                pending += 1
            else:
                other += 1
    return running, pending, other


def gpu_short_name(gpu_type: str) -> str:
    """nvidia-h200-141gb -> h200"""
    return gpu_type.replace("nvidia-", "").split("-")[0]


def make_job_name(base: str, dataset: str, gpu_type: str, region: str) -> str:
    """e.g. regen-gemma4-26b-perfectblend-h200-europe-west1"""
    return f"{base}-{dataset}-{gpu_short_name(gpu_type)}-{region}"


def patch_and_apply_yaml(
    job_yaml: str,
    base_name: str,
    dataset: str,
    gpu_type: str,
    region: str,
) -> Tuple[bool, str]:
    """Patch YAML and apply via kubectl. Returns (success, error_message)."""
    job_name = make_job_name(base_name, dataset, gpu_type, region)

    with open(job_yaml) as f:
        content = f.read()

    # Patch job name (first occurrence)
    content = content.replace(f"name: {base_name}", f"name: {job_name}", 1)

    # Patch DATASETS value and GPU type
    lines = content.splitlines()
    patched_lines = []
    patch_next_value = False
    for line in lines:
        if patch_next_value and "value:" in line:
            line = re.sub(r'value: ".*"', f'value: "{dataset}"', line)
            patch_next_value = False
        if "name: DATASETS" in line:
            patch_next_value = True
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


def delete_job(job_name: str) -> bool:
    """Delete a job, returns True if successful."""
    result = subprocess.run(
        ["kubectl", "delete", "job", job_name, "--ignore-not-found"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


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


def deploy_dataset(cfg: Config, dataset: str):
    """Deploy a single dataset across all (cluster x gpu_type) combos."""
    print()
    print("=" * 60)
    print(f"  Deploying: {dataset}")
    print(f"  GPU types: {', '.join(gpu_short_name(g) for g in cfg.gpu_types)}")
    print("=" * 60)

    # Phase 1: Submit to all combinations
    submitted: List[Tuple[str, str, str]] = []

    for gpu in cfg.gpu_types:
        for cluster, region in cfg.clusters:
            job_name = make_job_name(cfg.base_name, dataset, gpu, region)

            if not use_cluster(cluster, region, cfg.project):
                print(f"  {cluster}: failed to connect, skipping.")
                continue

            success, err = patch_and_apply_yaml(
                cfg.job_yaml,
                cfg.base_name,
                dataset,
                gpu,
                region,
            )
            if success:
                print(f"  {job_name}: submitted.")
                submitted.append((cluster, region, gpu))
            else:
                print(f"  {job_name}: kubectl apply failed, skipping.")
                for line in err.splitlines()[:5]:
                    print(f"    {line}")

    if not submitted:
        print(f"  Error: Failed to submit {dataset} to any cluster.")
        return

    # Phase 2: Wait for pods to schedule
    print(f"  Waiting up to {cfg.schedule_timeout}s for pods to schedule...")

    elapsed = 0
    winner: Optional[Tuple[str, str, str]] = None

    while elapsed < cfg.schedule_timeout:
        for cluster, region, gpu in submitted:
            job_name = make_job_name(cfg.base_name, dataset, gpu, region)

            if not use_cluster(cluster, region, cfg.project):
                continue

            running, pending, other = get_pod_status(job_name)

            if running >= cfg.min_running:
                winner = (cluster, region, gpu)
                break

        if winner:
            break

        print(f"  [{elapsed}s] No fully scheduled cluster yet...")
        time.sleep(30)
        elapsed += 30

    if not winner:
        print(
            f"  Warning: {dataset} not scheduled within "
            f"{cfg.schedule_timeout}s. Jobs remain submitted."
        )
        return

    # Phase 3: Keep winner, delete the rest
    win_cluster, win_region, win_gpu = winner
    winner_name = make_job_name(cfg.base_name, dataset, win_gpu, win_region)
    print(f"  Scheduled on: {winner_name}")

    for cluster, region, gpu in submitted:
        if (cluster, region, gpu) == winner:
            continue
        job_name = make_job_name(cfg.base_name, dataset, gpu, region)
        if use_cluster(cluster, region, cfg.project):
            delete_job(job_name)
            print(f"  {job_name}: deleted.")

    print(f"  {dataset} -> {winner_name} ({win_cluster}, {gpu_short_name(win_gpu)})")


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
    print("=" * 60)

    for ds in cfg.datasets:
        deploy_dataset(cfg, ds)

    print()
    print("=" * 60)
    print(f"  All {len(cfg.datasets)} dataset(s) dispatched.")
    print("=" * 60)


def cmd_build_and_redeploy(cfg: Config):
    """Build image, push, update YAML, delete old jobs, deploy new ones."""
    print()
    print("Building and pushing new image...")
    build_script = str(SCRIPT_DIR / "build_and_push.sh")
    result = subprocess.run(
        ["bash", build_script],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: Build failed.\n{result.stderr}")
        sys.exit(1)

    # Extract new tag from output
    new_tag = None
    for line in result.stdout.splitlines():
        if "Image:" in line:
            match = re.search(r"specforge-regen:(\S+)", line)
            if match:
                new_tag = match.group(1)

    if not new_tag:
        print("Error: Could not extract image tag from build output.")
        sys.exit(1)

    new_image = (
        f"us-central1-docker.pkg.dev/pyc-vtx-dev/"
        f"pyc-vtx-us-central1/specforge-regen:{new_tag}"
    )
    print(f"Updating YAML image to: {new_image}")

    with open(cfg.job_yaml) as f:
        content = f.read()

    content = re.sub(
        r"(image:\s*)(\S+)",
        rf"\g<1>{new_image}",
        content,
        count=1,
    )

    with open(cfg.job_yaml, "w") as f:
        f.write(content)

    print()
    cmd_delete(cfg)
    print()
    cmd_deploy(cfg)


def cmd_redeploy(cfg: Config):
    print()
    print("Redeploying (delete + deploy)...")
    cmd_delete(cfg)
    print()
    cmd_deploy(cfg)


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
        help="Comma-separated dataset names (default: magpie-multilingual,perfectblend)",
    )
    parser.add_argument(
        "--gpu-types",
        default="nvidia-h200-141gb,nvidia-h100-80gb",
        help="Comma-separated GPU types to try (default: nvidia-h200-141gb,nvidia-h100-80gb)",
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
        "--redeploy", action="store_true", help="Delete existing jobs and redeploy"
    )
    cmd_group.add_argument(
        "--build", action="store_true", help="Build image, push, update YAML, redeploy"
    )

    args = parser.parse_args()
    cfg = Config.from_args(args)

    if args.status:
        cmd_status(cfg)
    elif args.delete:
        cmd_delete(cfg)
    elif args.redeploy:
        cmd_redeploy(cfg)
    elif args.build:
        cmd_build_and_redeploy(cfg)
    else:
        cmd_deploy(cfg)


if __name__ == "__main__":
    main()
