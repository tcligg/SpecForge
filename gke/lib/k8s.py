"""kubectl wrappers used by the deploy CLI.

Step 2 extracted these from gke/deploy.py. Step 5 extends the module
with JSON-returning helpers (``get_pods_json``, ``get_pod_events``)
that the pending-reason classifier in ``gke/scheduling.py`` consumes.

The classifier needs structured pod state (conditions, container
statuses, scheduling events) rather than the loose stdout that
``get_pod_status`` returns; rather than re-parse ``kubectl get pods``
output we shell out with ``-o json`` and let the classifier work on
dicts. Tests can substitute captured JSON fixtures.

The single legacy rename kept from step 2: ``_wait_for_job_completion``
-> ``wait_for_job_completion``. ``gke/deploy.py`` re-exports the old
underscore name for any external caller.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Step 5 — JSON-returning helpers for the pending-reason classifier
# ---------------------------------------------------------------------------


def _get_json(cmd: List[str]) -> Any:
    """Run ``kubectl ... -o json`` and parse the result.

    Returns the parsed JSON document on success or ``None`` on any
    failure (non-zero exit, empty stdout, parse error). Callers must
    handle the ``None`` case; the classifier degrades to ``UNKNOWN``
    when pod or event data is unavailable, which matches the PLAN's
    "anything else" fallback.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def get_pods_json(job_name: str) -> List[Dict[str, Any]]:
    """Return the full list of pods for a job as parsed JSON dicts.

    Empty list on any failure. The classifier inspects
    ``pod['status']``, ``pod['spec']``, etc. so we deliberately return
    the raw kubectl shape rather than a narrowed view.
    """
    doc = _get_json(
        [
            "kubectl",
            "get",
            "pods",
            "-l",
            f"job-name={job_name}",
            "-o",
            "json",
        ]
    )
    if not doc or not isinstance(doc, dict):
        return []
    items = doc.get("items") or []
    return list(items) if isinstance(items, list) else []


def get_job_json(job_name: str) -> Optional[Dict[str, Any]]:
    """Return ``kubectl get job <name> -o json`` parsed; ``None`` on miss.

    Used by the orchestrator's adopt-existing-job check (step 6): if
    state recorded a Job and that Job still exists and is Active or
    Complete, we skip the submit step. ``None`` covers both
    "kubectl failed" and "job not found" — caller treats both as
    "submit fresh."
    """
    doc = _get_json(["kubectl", "get", "job", job_name, "-o", "json"])
    if not isinstance(doc, dict):
        return None
    return doc


def get_pod_events(pod_name: str) -> List[Dict[str, Any]]:
    """Return scheduling/lifecycle events for one pod as parsed JSON.

    Filters by ``involvedObject.name`` so each call's payload stays
    bounded to one pod's events. Empty list on any failure (and on
    pods that have no events yet, which is normal in the first few
    seconds after submission).
    """
    doc = _get_json(
        [
            "kubectl",
            "get",
            "events",
            "--field-selector",
            f"involvedObject.name={pod_name}",
            "-o",
            "json",
        ]
    )
    if not doc or not isinstance(doc, dict):
        return []
    items = doc.get("items") or []
    return list(items) if isinstance(items, list) else []
