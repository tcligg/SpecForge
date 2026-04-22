"""GKE cluster discovery and kubectl context switching.

Extracted verbatim from gke/deploy.py in step 2. No logic changes.
"""

from __future__ import annotations

import subprocess
import sys
from typing import List, Tuple


def discover_clusters(project: str) -> List[Tuple[str, str]]:
    """Auto-discover pyc-batch* clusters, excluding pyc-batch-us-south1."""
    # Local import: k8s.run_output is a more general helper but we
    # can't import it here without a circular dependency (k8s imports
    # use_cluster from this module). Inline the small subprocess call.
    print(f"Discovering pyc-batch* clusters in project {project}...")
    result = subprocess.run(
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
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() if result.returncode == 0 else ""

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
