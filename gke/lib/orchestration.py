"""Per-dataset deploy orchestration ("racing" pattern).

Extracted verbatim from gke/deploy.py in step 2. No logic changes.

This is the candidate-racing logic that step 6 of the rebuild plan
will replace with strict scheduling + serial fall-through. Keeping it
here as a focused module so the eventual deletion is one-shot.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from gke.lib.clusters import use_cluster
from gke.lib.config import Config
from gke.lib.k8s import delete_job, get_pod_status
from gke.lib.yaml_patch import gpu_short_name, make_job_name, patch_and_apply_yaml


def deploy_dataset(cfg: Config, dataset: str) -> Optional[Tuple[str, str, str]]:
    """
    Deploy a single dataset across all (cluster x gpu_type) combos.
    Returns the (job_name, cluster, region) of the winning job, or None.
    """
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
                cfg,
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
        return None

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
        return None

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
    return winner_name, win_cluster, win_region
