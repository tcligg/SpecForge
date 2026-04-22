"""Config dataclass shared by the deploy CLI and library helpers.

Extracted from gke/deploy.py in step 2. Lives here (rather than in
gke/deploy.py) so that lib modules can type-hint against it without
creating a circular import. The PLAN.md decision log records this as
the only deviation from the "Config stays in deploy.py" line in the
step 2 spec.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Resolved at import time so callers can derive sibling paths (the YAML
# files, build_and_push.sh, etc.) without re-deriving from __file__.
SCRIPT_DIR = Path(__file__).parent.parent.resolve()


@dataclass
class Config:
    project: str = "cloud-llm-test"
    schedule_timeout: int = 3600
    job_yaml: str = ""
    output_dir: Optional[str] = None
    datasets: List[str] = field(default_factory=list)
    gpu_types: List[str] = field(default_factory=list)
    clusters: List[Tuple[str, str]] = field(default_factory=list)
    base_name: str = ""
    total_pods: int = 8
    min_running: int = 4
    # Optional GCS URI (gs://bucket/path/state.json) where cmd_execute
    # writes a per-phase observer record. Step 3 only writes; step 4+
    # starts reading. None means state collection is disabled.
    state_uri: Optional[str] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        # Local import to avoid a top-level dependency on the clusters
        # module (and thus on subprocess / gcloud) when callers only
        # need to construct a Config from a fully-populated namespace.
        from gke.lib.clusters import discover_clusters

        cfg = cls()
        cfg.project = args.project
        cfg.schedule_timeout = args.timeout
        cfg.job_yaml = args.yaml
        cfg.output_dir = args.output_dir
        cfg.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        cfg.gpu_types = [g.strip() for g in args.gpu_types.split(",") if g.strip()]
        cfg.state_uri = getattr(args, "state_uri", None)

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
