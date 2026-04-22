"""kubectl wrappers used by the deploy CLI.

Extracted verbatim from gke/deploy.py in step 2. No logic changes.

The single rename: _wait_for_job_completion -> wait_for_job_completion.
gke/deploy.py re-exports the old underscore name for any external caller.
"""

from __future__ import annotations

import subprocess
import time
from typing import List, Tuple

from gke.lib.clusters import use_cluster
from gke.lib.config import Config


def run_output(cmd: List[str]) -> str:
    """Run a command and return stdout, empty string on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


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


def delete_job(job_name: str) -> bool:
    """Delete a job, returns True if successful."""
    result = subprocess.run(
        ["kubectl", "delete", "job", job_name, "--ignore-not-found"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def wait_for_job_completion(
    job_name: str, cluster: str, region: str, cfg: Config, timeout: int = 7200
) -> bool:
    """Polls a GKE job until it completes or fails."""
    if not use_cluster(cluster, region, cfg.project):
        print(f"Error: Could not switch to cluster {cluster} to monitor job.")
        return False

    elapsed = 0
    while elapsed < timeout:
        output = run_output(
            [
                "kubectl",
                "get",
                "job",
                job_name,
                "-o",
                "jsonpath='{.status.conditions[*].type}'",
            ]
        )
        # output is single-quoted, e.g. 'Complete' or 'Failed'
        status = output.strip("'")

        if "Complete" in status:
            print(f"  Job {job_name} completed successfully.")
            return True
        if "Failed" in status:
            print(f"  Error: Job {job_name} has failed.")
            return False

        print(f"  [{elapsed}s] Job {job_name} still running...")
        time.sleep(60)
        elapsed += 60

    print(f"  Error: Timeout waiting for job {job_name} to complete.")
    return False
