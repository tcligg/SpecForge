#!/usr/bin/env python3
"""Single-entrypoint orchestrator for the SpecForge GKE regen pipeline.

Steps 4a + 4b of gke/PLAN.md. The pipeline phases are::

    Phase A — PREPARE   (host)   prepare_data.py per dataset -> GCS
    Phase B — BUILD     (host)   build_and_push.sh, tag = git-sha
    Phase C — DEPLOY    (GKE)    submit job; wait for FULL parallelism
    Phase D — MONITOR   (GKE)    poll until terminal; per-shard status
    Phase D.5 — RESCUE  (GKE)    if partial: dispatch shard-aware rescue
    Phase E — MERGE     (host)   recursive gsutil compose to single jsonl

After step 4b lands, Phases A/B/E are native here. Phases C and D use
the helpers in ``gke.lib.{orchestration,k8s}`` directly (no shell-out
to ``gke/deploy.py --execute``); step 6 will replace the legacy
candidate-racing logic with strict scheduling + serial fall-through.
The deploy CLI is now used only for ``--status`` / ``--delete`` ops
shortcuts.

CLI shape::

    python3 gke/orchestrator.py run --config gke/regen-gemma3-27b.yaml \
        [--datasets a,b]               # override YAML's DATASETS env
        [--output-dir gs://bucket/p]   # override YAML's OUTPUT_DIR
        [--gpu-types h200,h100]        # passed to deploy lib
        [--clusters name:region,...]   # passed to deploy lib
        [--timeout 3600]               # per-dataset schedule timeout
        [--run-id ID]                  # resume an existing run
        [--new]                        # force a fresh run_id
        [--from prepare|build|deploy|monitor|merge]
        [--state-uri gs://...]         # override derived state path
        [--ignore-config-change]       # accept config_hash mismatch
        [--force-rebuild]              # bypass cached image in Phase B

    python3 gke/orchestrator.py status --config <yaml> [--run-id ID]
    python3 gke/orchestrator.py logs    ...   # stub
    python3 gke/orchestrator.py cancel  ...   # stub
    python3 gke/orchestrator.py cleanup ...   # stub
    python3 gke/orchestrator.py rescue  ...   # stub

Resume rules:

* ``run`` without ``--run-id`` always starts a new run. Resuming
  requires passing ``--run-id <id>`` explicitly. Auto-discovery by
  ``config_hash`` is deferred until step 6.
* The state URI defaults to
  ``gs://<bucket>/<output-prefix>/<run_id>/state.json`` derived from
  the YAML's ``OUTPUT_DIR`` env (``/gcs/<bucket>/<path>``).
  ``--state-uri`` overrides.
* Phase B skips when the computed image tag is already present in
  Artifact Registry (verified via ``gcloud artifacts docker images
  describe``). ``--force-rebuild`` bypasses the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make sure ``gke`` is importable when this file is invoked as a script
# from the repo root (``python3 gke/orchestrator.py ...``).
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gke import rescue as _rescue  # noqa: E402
from gke import state as _state  # noqa: E402
from gke.lib import merge as _merge  # noqa: E402
from gke.lib.clusters import discover_clusters, use_cluster  # noqa: E402
from gke.lib.config import Config as _DeployConfig  # noqa: E402
from gke.lib.k8s import get_job_json, wait_for_job_completion  # noqa: E402
from gke.lib.orchestration import (  # noqa: E402
    StrictDeployError,
    StrictDeployResult,
    deploy_dataset_strict,
)
from gke.lib.yaml_patch import gpu_short_name  # noqa: E402

PHASE_ORDER: Tuple[str, ...] = ("prepare", "build", "deploy", "monitor", "merge")

# Phase B build defaults. These mirror gke/build_and_push.sh's defaults
# so the orchestrator can decide *before* shelling out whether the
# resulting image already exists in GAR. Override via env or by editing
# the wrapper script and these constants together.
_BUILD_PROJECT = os.environ.get("PROJECT", "pyc-vtx-dev")
_BUILD_REGION = os.environ.get("REGION", "us-central1")
_BUILD_REPO = os.environ.get("REPO", "pyc-vtx-us-central1")
_BUILD_IMAGE_NAME = os.environ.get("IMAGE_NAME", "specforge-regen")


# ---------------------------------------------------------------------------
# YAML parsing (string-based, matches gke/lib/config.py's existing approach)
# ---------------------------------------------------------------------------


# Capture ``- name: KEY`` then a following ``value: "..."`` (or unquoted)
# without pulling in PyYAML. Mirrors the regex style already used in
# ``gke/lib/config.py`` so we stay consistent until step 8 swaps to
# ruamel.yaml. The DOTALL flag lets the regex span the newlines between
# the two keys; the lazy ``.*?`` keeps it from leaping past the next
# ``- name:`` block.
_ENV_BLOCK_RE_TEMPLATE = (
    r"-\s*name:\s*{key}\s*[\r\n]+\s*value:\s*\"?(?P<val>[^\"\r\n]+)\"?"
)


def _read_env_value(yaml_text: str, key: str) -> Optional[str]:
    """Pull a single ``env: -name: KEY / value: ...`` value out of YAML."""
    pat = re.compile(_ENV_BLOCK_RE_TEMPLATE.format(key=re.escape(key)))
    m = pat.search(yaml_text)
    if not m:
        return None
    return m.group("val").strip()


def _read_metadata_name(yaml_text: str) -> Optional[str]:
    m = re.search(r"^\s*name:\s*(\S+)", yaml_text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _gcs_uri_from_mount(path: str) -> Optional[str]:
    """Map a GCSFuse mount path back to its ``gs://`` URI.

    The YAMLs use ``OUTPUT_DIR: /gcs/<bucket>/<path>`` because the
    container sees the GCSFuse mount, but state and prepared inputs
    live at ``gs://<bucket>/<path>``. Returns None for paths we can't
    confidently map (caller falls back to explicit flags).
    """
    if not path:
        return None
    if path.startswith("gs://"):
        return path
    m = re.match(r"^/gcs/([^/]+)/?(.*)$", path)
    if not m:
        return None
    bucket, rest = m.group(1), m.group(2)
    rest = rest.rstrip("/")
    if rest:
        return f"gs://{bucket}/{rest}"
    return f"gs://{bucket}"


# ---------------------------------------------------------------------------
# OrchestratorConfig
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    """Run-time configuration for one orchestrator invocation.

    Distinct from ``gke.lib.config.Config`` (which is the deploy CLI's
    input). The orchestrator config carries everything needed to drive
    the full six-phase pipeline plus the state-machine metadata
    (``run_id``, ``state_uri``, resume flags). Phases B/C/D in step 4a
    delegate to the deploy CLI and reconstruct a ``Config`` there.
    """

    # Inputs ------------------------------------------------------------
    config_yaml: str
    datasets: List[str]
    output_dir: Optional[str]  # gs://... (host-side view; None to inherit YAML)
    project: str

    # Phase C/D inputs (forwarded to gke.lib.orchestration.deploy_dataset).
    # ``clusters_spec`` is the raw "name:region,name:region" string; an
    # empty string triggers auto-discover.
    gpu_types: List[str] = field(default_factory=list)
    clusters_spec: str = ""
    schedule_timeout: int = 3600

    # Run identity ------------------------------------------------------
    run_id: str = ""
    state_uri: str = ""

    # Resume / behaviour flags -----------------------------------------
    explicit_run_id: bool = False  # True if user passed --run-id (vs --new)
    from_phase: Optional[str] = None
    ignore_config_change: bool = False
    force_rebuild: bool = False
    # Step 5: forwarded to gke.lib.config.Config so deploy_dataset's
    # poll loop logs gke.scheduling.diagnose_job output. Step 6 makes
    # this the default.
    enable_scheduling_v2: bool = False

    # Step 6 — strict-scheduling tunables (forwarded to Config). See
    # gke/lib/config.py for semantics.
    attempt_deadline: int = 600
    monitor_deadline: int = 14400
    spot_exhausted_strategy: str = "sleep_retry"
    spot_retry_interval: int = 900
    max_wait_hours: int = 12

    # Step 7 — rescue (Phase D.5) tunables. PLAN.md:377.
    max_rescue_attempts: int = 3
    max_rescue_shards: int = 4

    # Derived from YAML (cached) ---------------------------------------
    base_name: str = ""
    yaml_output_dir: Optional[str] = None  # raw OUTPUT_DIR env (for reference)
    yaml_prepare_dir: Optional[str] = None  # PREPARE_DATA_DIR env

    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "OrchestratorConfig":
        config_yaml = args.config
        if not os.path.isfile(config_yaml):
            sys.exit(f"Error: --config {config_yaml!r} not found.")

        with open(config_yaml) as f:
            yaml_text = f.read()

        base_name = _read_metadata_name(yaml_text) or "regen"
        yaml_output_dir = _read_env_value(yaml_text, "OUTPUT_DIR")
        yaml_prepare_dir = _read_env_value(yaml_text, "PREPARE_DATA_DIR")
        yaml_datasets = _read_env_value(yaml_text, "DATASETS") or ""

        # Datasets: --datasets overrides; else fall back to YAML's env.
        ds_source = (args.datasets or yaml_datasets).strip()
        if not ds_source:
            sys.exit(
                "Error: no datasets to process. Pass --datasets a,b or set "
                "DATASETS in the YAML's env."
            )
        datasets = [d.strip() for d in ds_source.split(",") if d.strip()]

        # output_dir: --output-dir wins; else map YAML's OUTPUT_DIR to gs://.
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = _gcs_uri_from_mount(yaml_output_dir or "")
        if output_dir and not output_dir.startswith("gs://"):
            sys.exit(
                f"Error: --output-dir must be a gs:// URI, got {output_dir!r}. "
                "Pass --output-dir or fix the YAML's OUTPUT_DIR."
            )

        # Resume identity ----------------------------------------------
        new_flag = bool(getattr(args, "new", False))
        run_id_flag = getattr(args, "run_id", None)
        if run_id_flag and new_flag:
            sys.exit("Error: --run-id and --new are mutually exclusive.")
        if run_id_flag:
            run_id = run_id_flag
            explicit = True
        else:
            run_id = _state.compute_run_id(base_name)
            explicit = False
            if not new_flag:
                # No flag was passed; we default to a fresh run per the
                # step-4a decision and warn so the operator knows resume
                # requires --run-id.
                print(
                    "  Note: no --run-id given; starting a fresh run "
                    f"({run_id}). Pass --run-id to resume an existing one.",
                    file=sys.stderr,
                )

        # State URI ---------------------------------------------------
        state_uri = getattr(args, "state_uri", None) or ""
        if not state_uri:
            if not output_dir:
                sys.exit(
                    "Error: cannot derive --state-uri because OUTPUT_DIR is "
                    "missing from the YAML and --output-dir was not passed."
                )
            state_uri = f"{output_dir.rstrip('/')}/{run_id}/state.json"
        validated = _state.parse_state_uri(state_uri)
        if validated is None:
            sys.exit(
                f"Error: derived/declared state URI is not a valid GCS object: "
                f"{state_uri!r}"
            )
        state_uri = validated

        # --from validation ------------------------------------------
        from_phase = getattr(args, "from_phase", None)
        if from_phase is not None and from_phase not in PHASE_ORDER:
            sys.exit(f"Error: --from must be one of {PHASE_ORDER}, got {from_phase!r}")

        gpu_types_raw = getattr(args, "gpu_types", None) or ""
        gpu_types = [g.strip() for g in gpu_types_raw.split(",") if g.strip()]
        clusters_spec = getattr(args, "clusters", None) or ""

        return cls(
            config_yaml=os.path.abspath(config_yaml),
            datasets=datasets,
            output_dir=output_dir,
            project=args.project,
            gpu_types=gpu_types,
            clusters_spec=clusters_spec,
            schedule_timeout=int(getattr(args, "timeout", 3600) or 3600),
            run_id=run_id,
            state_uri=state_uri,
            explicit_run_id=explicit,
            from_phase=from_phase,
            ignore_config_change=bool(getattr(args, "ignore_config_change", False)),
            force_rebuild=bool(getattr(args, "force_rebuild", False)),
            enable_scheduling_v2=bool(getattr(args, "enable_scheduling_v2", False)),
            attempt_deadline=int(getattr(args, "attempt_deadline", 600) or 600),
            monitor_deadline=int(getattr(args, "monitor_deadline", 14400) or 14400),
            spot_exhausted_strategy=str(
                getattr(args, "spot_exhausted_strategy", "sleep_retry") or "sleep_retry"
            ),
            spot_retry_interval=int(getattr(args, "spot_retry_interval", 900) or 900),
            max_wait_hours=int(getattr(args, "max_wait_hours", 12) or 12),
            max_rescue_attempts=int(getattr(args, "max_rescue_attempts", 3) or 3),
            max_rescue_shards=int(getattr(args, "max_rescue_shards", 4) or 4),
            base_name=base_name,
            yaml_output_dir=yaml_output_dir,
            yaml_prepare_dir=yaml_prepare_dir,
        )


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _open_or_create_state(
    cfg: OrchestratorConfig,
) -> Tuple[_state.StateStore, _state.RunState]:
    """Get a (store, state) pair, creating the GCS object if missing.

    Behaviour:

    * If the GCS object exists: read it. Verify ``config_hash`` matches
      the current YAML; abort unless ``--ignore-config-change`` is set.
      Verify ``run_id`` matches; if not, the operator passed
      ``--run-id`` for a different run living at the same URI — abort.
    * If the object does not exist: write a fresh ``initial_state``.
    """
    store = _state.StateStore(cfg.state_uri)
    config_hash = _state.compute_config_hash(cfg.config_yaml)

    existing = store.read()
    if existing is None:
        if cfg.explicit_run_id:
            print(
                f"  Note: --run-id {cfg.run_id} given but no state object "
                f"exists at {cfg.state_uri}; creating a new one."
            )
        s = _state.initial_state(
            run_id=cfg.run_id,
            config_yaml=cfg.config_yaml,
            config_hash=config_hash,
        )
        store.write(s)
        return store, s

    # Existing run -----------------------------------------------------
    if existing.run_id != cfg.run_id:
        sys.exit(
            f"Error: state at {cfg.state_uri} has run_id={existing.run_id!r}, "
            f"but the orchestrator was invoked with run_id={cfg.run_id!r}. "
            "Use --new for a fresh run, or pass the matching --run-id."
        )
    if existing.config_hash != config_hash and not cfg.ignore_config_change:
        sys.exit(
            "Error: config_hash mismatch — the YAML changed since this run "
            "was created.\n"
            f"  state: {existing.config_hash}\n"
            f"  yaml:  {config_hash}\n"
            "Pass --ignore-config-change to proceed anyway."
        )
    return store, existing


def _phase_status(
    state: _state.RunState, phase: str, *, datasets: Sequence[str]
) -> str:
    """Summarise a phase as ``ok|partial|none`` for status output."""
    phases = state.phases
    if phase == "build":
        return "ok" if phases.build.image_uri else "none"
    bag: Dict[str, Dict] = getattr(phases, phase)
    if not bag:
        return "none"
    ok = 0
    for ds in datasets:
        rec = bag.get(ds) or {}
        # Each phase records its own happy-path marker; we treat any
        # non-empty record as "started" and require status==ok or a
        # known terminal field for "ok".
        if rec.get("status") == "ok":
            ok += 1
        elif phase == "monitor" and rec.get("primary_job_status") == "Complete":
            ok += 1
        elif phase == "deploy" and rec.get("primary_job"):
            ok += 1  # at least submitted; finer state arrives in step 6
    if ok == 0:
        return "none"
    if ok == len(datasets):
        return "ok"
    return "partial"


# ---------------------------------------------------------------------------
# Phase A — prepare (native implementation)
# ---------------------------------------------------------------------------


def _gsutil_stat_size(uri: str) -> Optional[int]:
    """Return the object's content length in bytes, or None if missing."""
    proc = subprocess.run(
        ["gsutil", "stat", uri],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    m = re.search(r"^\s*Content-Length:\s*(\d+)\s*$", proc.stdout, re.MULTILINE)
    return int(m.group(1)) if m else None


def _count_jsonl_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _prepare_destination(cfg: OrchestratorConfig, dataset: str) -> str:
    """GCS URI for the prepared ``<dataset>_train.jsonl`` artifact.

    Today, prepared data lives at ``gs://pyc-eagle-data/datasets/prepared/``
    per the YAML's ``PREPARE_DATA_DIR`` env (a /gcs mount). We honour that
    so jobs that read from the same path see what we just uploaded.
    """
    base_mount = cfg.yaml_prepare_dir or "/gcs/pyc-eagle-data/datasets/prepared"
    base_uri = _gcs_uri_from_mount(base_mount)
    if not base_uri:
        sys.exit(
            f"Error: PREPARE_DATA_DIR={base_mount!r} is not a recognized "
            "GCSFuse mount; cannot derive a gs:// destination."
        )
    return f"{base_uri.rstrip('/')}/{dataset}_train.jsonl"


def _record_prepare(
    store: _state.StateStore,
    dataset: str,
    *,
    status: str,
    uri: str,
    size: int,
    n_lines: int,
) -> None:
    def _mut(s: _state.RunState) -> _state.RunState:
        s.phases.prepare[dataset] = {
            "status": status,
            "uri": uri,
            "size": size,
            "n_lines": n_lines,
            "completed_at": _state._utc_now_iso(),
        }
        return s

    store.update(_mut)


def phase_a_prepare(
    cfg: OrchestratorConfig,
    store: _state.StateStore,
    state: _state.RunState,
    *,
    force: bool = False,
) -> None:
    """Prepare each dataset locally and publish to GCS.

    Skip rules:
      * ``state.phases.prepare[ds].status == "ok"`` AND
      * the recorded ``uri`` exists in GCS with size >= recorded size.

    On skip, we still re-stat to confirm the object hasn't been deleted.
    On miss, run ``scripts/prepare_data.py`` into a tempdir and ``gsutil
    cp`` the result into place. Line count is recorded for Phase D.5
    (rescue) and for sanity-checking the merged output in Phase E.
    """
    print()
    print("[Phase A] Prepare datasets")
    print("-" * 60)

    prepare_script = _PROJECT_ROOT / "scripts" / "prepare_data.py"
    if not prepare_script.is_file():
        sys.exit(f"Error: {prepare_script} not found.")

    for dataset in cfg.datasets:
        dest_uri = _prepare_destination(cfg, dataset)
        rec = state.phases.prepare.get(dataset) or {}
        recorded_uri = rec.get("uri")
        recorded_size = int(rec.get("size") or 0)

        if not force and rec.get("status") == "ok" and recorded_uri == dest_uri:
            actual = _gsutil_stat_size(dest_uri)
            if actual is not None and actual >= recorded_size > 0:
                print(
                    f"  [skip] {dataset}: already prepared at {dest_uri} "
                    f"(size={actual})"
                )
                continue
            print(
                f"  [redo] {dataset}: state says ok but GCS object missing "
                f"or smaller than recorded (recorded={recorded_size}, "
                f"actual={actual})"
            )

        print(f"  [run]  {dataset}: preparing -> {dest_uri}")
        with tempfile.TemporaryDirectory(prefix=f"prepare-{dataset}-") as tmp:
            local_dir = Path(tmp)
            cmd = [
                sys.executable,
                str(prepare_script),
                "--dataset",
                dataset,
                "--output-path",
                str(local_dir),
            ]
            print(f"         $ {' '.join(cmd)}")
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                sys.exit(
                    f"Error: prepare_data.py failed for {dataset} "
                    f"(exit {proc.returncode})"
                )

            local_path = local_dir / f"{dataset}_train.jsonl"
            if not local_path.is_file():
                sys.exit(
                    f"Error: prepare_data.py produced no {local_path.name} "
                    f"in {local_dir}"
                )
            size = local_path.stat().st_size
            n_lines = _count_jsonl_lines(local_path)
            print(f"         local: {local_path} ({size} bytes, {n_lines} lines)")

            cp = subprocess.run(["gsutil", "cp", str(local_path), dest_uri])
            if cp.returncode != 0:
                sys.exit(f"Error: gsutil cp failed for {dataset}")

        _record_prepare(
            store,
            dataset,
            status="ok",
            uri=dest_uri,
            size=size,
            n_lines=n_lines,
        )
        print(f"  [done] {dataset}: uploaded ({size} bytes, {n_lines} lines)")


# ---------------------------------------------------------------------------
# Phase B — build (native)
# ---------------------------------------------------------------------------


def _git_head_sha(short: bool = True) -> str:
    """Return ``git rev-parse HEAD``; abort on failure."""
    args = (
        ["git", "rev-parse", "--short=12", "HEAD"]
        if short
        else ["git", "rev-parse", "HEAD"]
    )
    proc = subprocess.run(args, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    if proc.returncode != 0:
        sys.exit(f"Error: `{' '.join(args)}` failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _git_dirty_diff_hash() -> Optional[str]:
    """Return sha256[:8] of the working-tree diff against HEAD, or None if clean.

    "Dirty" means tracked-file modifications OR staged changes; untracked
    files are deliberately ignored (they'd otherwise add noise from local
    artifacts like the sample jsonl files in the workspace root). This
    matches the spirit of PLAN.md's tagging rule while keeping the tag
    stable across `python3 ...` runs that drop pyc files.
    """
    proc = subprocess.run(
        ["git", "diff", "HEAD"],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    if proc.returncode != 0:
        # Treat git failure conservatively: act as if clean. The git-sha
        # tag still uniquely identifies HEAD; we just lose the dirty
        # marker. Operator can pass --force-rebuild if needed.
        return None
    diff = proc.stdout
    if not diff.strip():
        return None
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()[:8]


def _compute_image_tag(force_rebuild: bool) -> Tuple[str, str, bool]:
    """Compute (tag, git_sha, dirty). ``force_rebuild`` appends a salt.

    PLAN.md:262 calls for ``<git-sha>`` clean / ``<git-sha>-dirty<hash>``
    dirty. ``--force-rebuild`` adds a wall-clock suffix so we always get
    a fresh tag (otherwise GAR-describe would still report the tag as
    present and we'd skip).
    """
    sha = _git_head_sha(short=True)
    diff_hash = _git_dirty_diff_hash()
    dirty = diff_hash is not None
    tag = sha if not dirty else f"{sha}-dirty{diff_hash}"
    if force_rebuild:
        import time as _time

        tag = f"{tag}-force{int(_time.time())}"
    return tag, sha, dirty


def _full_image_uri(tag: str) -> str:
    return (
        f"{_BUILD_REGION}-docker.pkg.dev/"
        f"{_BUILD_PROJECT}/{_BUILD_REPO}/{_BUILD_IMAGE_NAME}:{tag}"
    )


def _gar_image_exists(image_uri: str) -> bool:
    """Returns True iff ``gcloud artifacts docker images describe`` succeeds.

    We discard stderr; the only signal we want is the exit code. A
    successful describe means the image+tag is already published, so
    Phase B can skip the build/push.
    """
    proc = subprocess.run(
        ["gcloud", "artifacts", "docker", "images", "describe", image_uri],
        capture_output=True,
    )
    return proc.returncode == 0


def _patch_yaml_image(yaml_path: str, image_uri: str) -> None:
    """Rewrite the YAML's first non-comment ``image:`` line to ``image_uri``.

    Tightened from the legacy ``deploy.py:cmd_execute`` substitution
    (which used ``(image:\\s*)(\\S+)`` with ``count=1``) because the
    unpinned YAMLs now carry an explanatory comment containing the
    word ``image:``; we must skip comment lines and only target the
    real key.
    """
    with open(yaml_path) as f:
        content = f.read()
    # ``(?m)^`` anchors per-line; the leading group requires whitespace
    # then ``image:`` (no ``#`` before it), and we replace only the
    # first such match.
    new_content, n = re.subn(
        r"(?m)^(?P<lead>[ \t]+image:[ \t]*)\S+",
        rf"\g<lead>{image_uri}",
        content,
        count=1,
    )
    if n == 0:
        sys.exit(
            f"Error: no `image:` key found in {yaml_path}; "
            "Phase B cannot pin the resolved image."
        )
    if new_content != content:
        with open(yaml_path, "w") as f:
            f.write(new_content)


def _record_build(
    store: _state.StateStore,
    *,
    image_uri: str,
    git_sha: str,
    dirty: bool,
) -> None:
    def _mut(s: _state.RunState) -> _state.RunState:
        s.phases.build = _state.BuildPhase(
            image_uri=image_uri,
            git_sha=git_sha,
            dirty=dirty,
            completed_at=_state._utc_now_iso(),
        )
        return s

    store.update(_mut)


def phase_b_build(
    cfg: OrchestratorConfig,
    store: _state.StateStore,
    state: _state.RunState,
) -> str:
    """Build (or skip) the container image, pin it into the YAML.

    Returns the resolved image URI; the caller is expected to use the
    pinned YAML for Phase C. State records ``image_uri``, ``git_sha``,
    and ``dirty``.

    Skip rules (in order):
      1. ``state.phases.build.image_uri`` matches the would-be tag and
         the image still exists in GAR -> skip build, pin YAML.
      2. ``--force-rebuild`` is unset and GAR already has the tag ->
         skip build, pin YAML, record state (in case prior run crashed
         before the state write).
      3. Otherwise -> shell out to ``gke/build_and_push.sh <tag>``.
    """
    print()
    print("[Phase B] Build & push image")
    print("-" * 60)

    tag, git_sha, dirty = _compute_image_tag(cfg.force_rebuild)
    image_uri = _full_image_uri(tag)
    print(f"  Tag:       {tag}")
    print(f"  Image:     {image_uri}")
    print(f"  Git SHA:   {git_sha}{' (dirty)' if dirty else ''}")

    recorded = state.phases.build.image_uri
    if recorded == image_uri and not cfg.force_rebuild:
        if _gar_image_exists(image_uri):
            print(f"  [skip] state already records this image and GAR confirms it.")
            _patch_yaml_image(cfg.config_yaml, image_uri)
            return image_uri
        print(f"  [redo] state records this image but GAR no longer has it.")

    if not cfg.force_rebuild and _gar_image_exists(image_uri):
        print(f"  [skip] image already in GAR; recording in state and pinning YAML.")
        _patch_yaml_image(cfg.config_yaml, image_uri)
        _record_build(store, image_uri=image_uri, git_sha=git_sha, dirty=dirty)
        return image_uri

    print(f"  [build] not in GAR (or --force-rebuild); shelling out.")
    build_script = _PROJECT_ROOT / "gke" / "build_and_push.sh"
    if not build_script.is_file():
        sys.exit(f"Error: {build_script} not found.")
    proc = subprocess.run(
        ["bash", str(build_script), tag],
        cwd=_PROJECT_ROOT,
        env={
            **os.environ,
            "PROJECT": _BUILD_PROJECT,
            "REGION": _BUILD_REGION,
            "REPO": _BUILD_REPO,
            "IMAGE_NAME": _BUILD_IMAGE_NAME,
        },
    )
    if proc.returncode != 0:
        sys.exit(f"Error: build_and_push.sh failed (exit {proc.returncode})")

    _patch_yaml_image(cfg.config_yaml, image_uri)
    _record_build(store, image_uri=image_uri, git_sha=git_sha, dirty=dirty)
    print(f"  [done] image pushed and pinned into {cfg.config_yaml}")
    return image_uri


# ---------------------------------------------------------------------------
# Phase C/D — deploy + monitor (using gke.lib helpers)
# ---------------------------------------------------------------------------


def _build_deploy_config(cfg: OrchestratorConfig) -> _DeployConfig:
    """Construct a ``gke.lib.config.Config`` for the deploy lib helpers.

    The legacy ``Config.from_args`` path expects an argparse Namespace
    and re-parses the YAML. We bypass that and build the dataclass
    directly so the orchestrator stays the single source of truth for
    inputs (datasets, output_dir, gpu_types, clusters).
    """
    if cfg.clusters_spec:
        clusters: List[Tuple[str, str]] = []
        for entry in cfg.clusters_spec.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            name, region = entry.split(":", 1)
            clusters.append((name.strip(), region.strip()))
    else:
        clusters = discover_clusters(cfg.project)

    if not cfg.gpu_types:
        sys.exit(
            "Error: --gpu-types is empty. Pass e.g. "
            "--gpu-types nvidia-h200-141gb,nvidia-h100-80gb"
        )

    deploy_cfg = _DeployConfig(
        project=cfg.project,
        schedule_timeout=cfg.schedule_timeout,
        job_yaml=cfg.config_yaml,
        output_dir=cfg.output_dir,
        datasets=list(cfg.datasets),
        gpu_types=list(cfg.gpu_types),
        clusters=clusters,
        base_name=cfg.base_name,
        state_uri=cfg.state_uri,
        enable_scheduling_v2=cfg.enable_scheduling_v2,
        attempt_deadline=cfg.attempt_deadline,
        monitor_deadline=cfg.monitor_deadline,
        spot_exhausted_strategy=cfg.spot_exhausted_strategy,
        spot_retry_interval=cfg.spot_retry_interval,
        max_wait_hours=cfg.max_wait_hours,
        max_rescue_attempts=cfg.max_rescue_attempts,
        max_rescue_shards=cfg.max_rescue_shards,
    )
    # total_pods comes from the YAML's ``completions:``. Step 6 sets
    # ``min_running = total_pods`` (strict default; PLAN.md:323) so
    # deploy_dataset_strict only adopts a candidate when every shard
    # is Running. The legacy ceil(N/2) heuristic was the source of
    # the silent-partial-completion data loss this rebuild fixes.
    #
    # CHUNK_SIZE comes from the YAML's env (default 500) so the rescue
    # path knows the chunk-id space without re-reading the YAML.
    with open(cfg.config_yaml) as f:
        ymtext = f.read()
    m = re.search(r"completions:\s*(\d+)", ymtext)
    deploy_cfg.total_pods = int(m.group(1)) if m else 8
    deploy_cfg.min_running = deploy_cfg.total_pods
    chunk_size_str = _read_env_value(ymtext, "CHUNK_SIZE")
    if chunk_size_str:
        try:
            deploy_cfg.chunk_size = int(chunk_size_str)
        except ValueError:
            pass  # fall back to the dataclass default (500)
    return deploy_cfg


def _record_deploy(
    store: _state.StateStore,
    dataset: str,
    *,
    job_name: str,
    cluster: str,
    region: str,
    gpu: str,
    total_shards: int,
) -> None:
    def _mut(s: _state.RunState) -> _state.RunState:
        s.phases.deploy[dataset] = {
            "primary_job": {
                "cluster": cluster,
                "region": region,
                "gpu": gpu,
                "job_name": job_name,
                "total_shards": total_shards,
                "started_at": _state._utc_now_iso(),
            }
        }
        return s

    store.update(_mut)


def _record_monitor(
    store: _state.StateStore,
    dataset: str,
    *,
    success: bool,
) -> None:
    def _mut(s: _state.RunState) -> _state.RunState:
        s.phases.monitor[dataset] = {
            "primary_job_status": "Complete" if success else "Failed",
            # Per-shard status arrives in step 6 once we read
            # .status.completedIndexes; record empty for now.
            "shard_status": {},
        }
        return s

    store.update(_mut)


def _job_status(job_doc: Dict[str, Any]) -> str:
    """Extract a coarse status from a kubectl Job JSON document.

    Returns one of ``"Complete"``, ``"Failed"``, ``"Active"``, or
    ``"Unknown"``. Used by the adopt-existing-job check so we can
    decide whether to re-submit.
    """
    status = job_doc.get("status") or {}
    conditions = status.get("conditions") or []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        if cond.get("status") != "True":
            continue
        ctype = cond.get("type")
        if ctype in ("Complete", "Failed"):
            return ctype  # type: ignore[return-value]
    if status.get("active"):
        return "Active"
    return "Unknown"


def _adopt_existing_job(
    cfg: OrchestratorConfig,
    deploy_cfg: _DeployConfig,
    dataset: str,
    state: _state.RunState,
) -> Optional[Tuple[str, str, str, str, str]]:
    """If state already records a Job that's still alive, adopt it.

    Returns ``(job_name, cluster, region, gpu, status)`` on adopt or
    ``None`` if the recorded job is missing/Failed (caller submits
    fresh) or no record exists.

    PLAN.md:345-347: this prevents accidentally double-submitting on
    a resume. Status ``"Complete"`` is the cheap path — Phase D
    immediately reports success without waiting.
    """
    rec = (state.phases.deploy.get(dataset) or {}).get("primary_job") or {}
    job_name = rec.get("job_name")
    cluster = rec.get("cluster")
    region = rec.get("region")
    gpu = rec.get("gpu") or ""
    if not (job_name and cluster and region):
        return None

    if not use_cluster(cluster, region, cfg.project):
        print(
            f"  [adopt] {dataset}: recorded job {job_name} on {cluster} but "
            f"kubectl context switch failed; submitting fresh."
        )
        return None

    job_doc = get_job_json(job_name)
    if job_doc is None:
        print(
            f"  [adopt] {dataset}: recorded job {job_name} no longer exists "
            f"on {cluster}; submitting fresh."
        )
        return None

    status = _job_status(job_doc)
    if status == "Failed":
        print(
            f"  [adopt] {dataset}: recorded job {job_name} is Failed; submitting fresh."
        )
        return None
    if status not in ("Active", "Complete"):
        print(
            f"  [adopt] {dataset}: recorded job {job_name} has unknown status; "
            f"submitting fresh."
        )
        return None

    print(f"  [adopt] {dataset}: reusing existing job {job_name} (status={status}).")
    return job_name, cluster, region, gpu, status


def phase_c_deploy_dataset(
    cfg: OrchestratorConfig,
    deploy_cfg: _DeployConfig,
    store: _state.StateStore,
    state: _state.RunState,
    dataset: str,
) -> Optional[Tuple[str, str, str, str]]:
    """Strict-scheduling Phase C: returns (job_name, cluster, region, gpu).

    1. Adopt-existing-job: if state already points at an Active or
       Complete Job, reuse it without resubmitting.
    2. Otherwise, drive ``deploy_dataset_strict`` and record the
       outcome. Operator-fixable failures (CONFIG_ERROR, IMAGE_PULL_ERROR)
       propagate as ``StrictDeployError`` so the caller can stop the
       whole pipeline rather than silently move to the next dataset.
    """
    adopted = _adopt_existing_job(cfg, deploy_cfg, dataset, state)
    if adopted is not None:
        job_name, cluster, region, gpu, _status = adopted
        return job_name, cluster, region, gpu

    try:
        result: StrictDeployResult = deploy_dataset_strict(deploy_cfg, dataset)
    except StrictDeployError as exc:
        print(f"  Error: {exc}")
        # Re-raise so ``cmd_run`` can decide whether to abort the
        # whole run; the existing per-dataset try/except in the
        # orchestrator catches this in step 6's wiring.
        raise

    if not result.succeeded:
        if result.spot_exhausted:
            print(
                f"  Warning: {dataset} spot-exhausted across all candidates "
                f"and --max-wait-hours hit; deferring."
            )
        return None

    assert result.winner is not None
    job_name, cluster, region = result.winner
    gpu = result.winner_gpu

    _record_deploy(
        store,
        dataset,
        job_name=job_name,
        cluster=cluster,
        region=region,
        gpu=gpu,
        total_shards=deploy_cfg.total_pods,
    )
    return job_name, cluster, region, gpu


def phase_d_monitor_dataset(
    deploy_cfg: _DeployConfig,
    store: _state.StateStore,
    dataset: str,
    *,
    job_name: str,
    cluster: str,
    region: str,
) -> bool:
    """Block until the job completes; record state. Returns success."""
    print(f"  Monitoring job {job_name} in {cluster}...")
    success = wait_for_job_completion(job_name, cluster, region, deploy_cfg)
    _record_monitor(store, dataset, success=success)
    return success


# ---------------------------------------------------------------------------
# Phase E — merge (native)
# ---------------------------------------------------------------------------


def _record_merge(
    store: _state.StateStore,
    dataset: str,
    result: _merge.MergeResult,
) -> None:
    def _mut(s: _state.RunState) -> _state.RunState:
        rec = result.to_state_dict()
        rec["completed_at"] = _state._utc_now_iso()
        rec["status"] = (
            "ok" if result.verified or result.expected_lines is None else "unverified"
        )
        s.phases.merge[dataset] = rec
        return s

    store.update(_mut)


def phase_e_merge_dataset(
    deploy_cfg: _DeployConfig,
    store: _state.StateStore,
    dataset: str,
) -> Optional[_merge.MergeResult]:
    """Merge per-chunk outputs into a single jsonl, record state.

    Uses ``gke.lib.merge.compose_dataset_chunks`` (recursive compose,
    line-count verification). When ``expected_lines`` and
    ``actual_lines`` disagree, the merge record is marked
    ``status="unverified"`` instead of ``"ok"`` so an operator can
    pick it out of state without re-running anything.
    """
    print(f"  Job {deploy_cfg.base_name}/{dataset} completed. Merging output chunks...")
    try:
        result = _merge.compose_dataset_chunks(deploy_cfg, dataset)
    except RuntimeError as exc:
        print(f"  Error: merge failed for {dataset}: {exc}")
        return None
    if result is None:
        return None
    _record_merge(store, dataset, result)
    return result


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _phases_to_run(cfg: OrchestratorConfig) -> List[str]:
    if cfg.from_phase is None:
        return list(PHASE_ORDER)
    start = PHASE_ORDER.index(cfg.from_phase)
    return list(PHASE_ORDER[start:])


def cmd_run(cfg: OrchestratorConfig) -> None:
    print()
    print("=" * 60)
    print("  GKE Regen Orchestrator")
    print("=" * 60)
    print(f"  Config:    {cfg.config_yaml}")
    print(f"  Run ID:    {cfg.run_id}")
    print(f"  State:     {cfg.state_uri}")
    print(f"  Datasets:  {', '.join(cfg.datasets)}")
    print(f"  Output:    {cfg.output_dir or '(inherits YAML)'}")
    if cfg.gpu_types:
        print(f"  GPU types: {', '.join(gpu_short_name(g) for g in cfg.gpu_types)}")
    if cfg.from_phase:
        print(f"  Resume:    from phase '{cfg.from_phase}'")
    print("=" * 60)

    store, state = _open_or_create_state(cfg)
    phases = _phases_to_run(cfg)

    if "prepare" in phases:
        phase_a_prepare(cfg, store, state)

    if "build" in phases:
        phase_b_build(cfg, store, state)
    elif state.phases.build.image_uri:
        # Resuming past Phase B: re-pin the recorded image into the YAML
        # so Phase C's kubectl apply uses the right tag. State is the
        # source of truth here; the YAML on disk could have been touched
        # by a different run.
        _patch_yaml_image(cfg.config_yaml, state.phases.build.image_uri)

    needs_cd = "deploy" in phases or "monitor" in phases
    needs_e = "merge" in phases
    if not (needs_cd or needs_e):
        _print_run_done(cfg)
        return

    deploy_cfg = _build_deploy_config(cfg)

    for ds in cfg.datasets:
        # Re-read state in case a prior dataset's update raced with us.
        # ``store.update`` retries on conflict so values stay coherent.
        latest = store.read() or state

        if needs_cd:
            existing_monitor = latest.phases.monitor.get(ds) or {}
            already_complete = existing_monitor.get("primary_job_status") == "Complete"
            if cfg.from_phase in (None, "prepare", "build") and already_complete:
                # Pure resume case: monitor record already terminal.
                # Nothing to redo on C/D; fall through to Phase E.
                print(f"  [skip] {ds}: monitor record already says Complete.")
            else:
                try:
                    deploy_result = phase_c_deploy_dataset(
                        cfg, deploy_cfg, store, latest, ds
                    )
                except StrictDeployError as exc:
                    # Operator-fixable failure: fail the pipeline rather
                    # than silently skip and lose the dataset. The
                    # remaining datasets are likely to hit the same
                    # error (image is global, YAML configs are shared).
                    sys.exit(f"Error: pipeline aborted: {exc}")
                if deploy_result is None:
                    print(
                        f"  Warning: failed to deploy {ds}; skipping monitor + merge."
                    )
                    continue
                job_name, cluster, region, _gpu = deploy_result
                ok = phase_d_monitor_dataset(
                    deploy_cfg,
                    store,
                    ds,
                    job_name=job_name,
                    cluster=cluster,
                    region=region,
                )
                # Phase D.5 (rescue): triggered when the primary job
                # didn't terminate cleanly *or* when chunks are still
                # missing despite a Complete status (defensive — the
                # scheduler has been observed to mark Jobs Complete
                # while a few shards quietly never produced output).
                latest = store.read() or state
                try:
                    missing = _rescue.compute_missing_chunks(latest, deploy_cfg, ds)
                except ValueError as exc:
                    print(f"  Warning: rescue analysis skipped for {ds}: {exc}")
                    missing = []
                if missing or not ok:
                    if not missing and not ok:
                        # Nothing to rescue but D failed — likely a
                        # whole-job failure with no chunks done yet.
                        # Skip rescue attempt for this dataset (it
                        # would just hit the same failure mode) and
                        # let the operator decide.
                        print(
                            f"  Error: {ds} job failed and no chunks done; "
                            f"skipping merge."
                        )
                        continue
                    if missing:
                        print(
                            f"  {ds}: {len(missing)} chunks missing; "
                            f"invoking Phase D.5 rescue."
                        )
                        try:
                            _rescue.phase_d5_rescue_dataset(
                                deploy_cfg,
                                latest,
                                store,
                                ds,
                                src_yaml=cfg.config_yaml,
                                base_name=cfg.base_name,
                            )
                        except _rescue.RescueError as exc:
                            print(f"  Error: rescue failed for {ds}: {exc}")
                            print(f"  Skipping merge for {ds}.")
                            continue

        if needs_e:
            phase_e_merge_dataset(deploy_cfg, store, ds)

    _print_run_done(cfg)


def _print_run_done(cfg: OrchestratorConfig) -> None:
    print()
    print("=" * 60)
    print("  Orchestrator run complete.")
    print(f"  State written to: {cfg.state_uri}")
    print("=" * 60)


def cmd_status(cfg: OrchestratorConfig) -> None:
    store = _state.StateStore(cfg.state_uri)
    state = store.read()
    if state is None:
        print(f"No state object at {cfg.state_uri}")
        return
    print(f"Run ID:        {state.run_id}")
    print(f"Config YAML:   {state.config_yaml}")
    print(f"Config hash:   {state.config_hash}")
    print(f"Created:       {state.created_at}")
    print(f"Updated:       {state.updated_at}")
    print()
    print("Phases:")
    for phase in PHASE_ORDER:
        summary = _phase_status(state, phase, datasets=cfg.datasets)
        print(f"  {phase:<10} {summary}")
    print()
    if state.phases.build.image_uri:
        b = state.phases.build
        print(
            f"Build:         {b.image_uri}  (sha={b.git_sha}"
            f"{', dirty' if b.dirty else ''})"
        )
        print()
    print("Per-dataset prepare:")
    for ds in cfg.datasets:
        rec = state.phases.prepare.get(ds) or {}
        print(
            f"  {ds:<28} status={rec.get('status', '-'):<6} "
            f"size={rec.get('size', '-')}  uri={rec.get('uri', '-')}"
        )
    print()
    print("Per-dataset deploy/monitor:")
    for ds in cfg.datasets:
        d = (state.phases.deploy.get(ds) or {}).get("primary_job") or {}
        m = state.phases.monitor.get(ds) or {}
        print(
            f"  {ds:<28} job={d.get('job_name', '-'):<48} "
            f"status={m.get('primary_job_status', '-')}"
        )
    print()
    print("Per-dataset merge:")
    for ds in cfg.datasets:
        rec = state.phases.merge.get(ds) or {}
        print(
            f"  {ds:<28} status={rec.get('status', '-'):<10} "
            f"chunks={rec.get('chunks_merged', '-')} "
            f"verified={rec.get('verified', '-')}  uri={rec.get('uri', '-')}"
        )


def _stub(name: str):
    def _impl(cfg: OrchestratorConfig) -> None:
        sys.exit(
            f"`{name}` is a stub. It will land in a later step of "
            "gke/PLAN.md. Use the underlying tools in the meantime "
            "(`kubectl`, `gke/deploy.py --status`, `gsutil rm`)."
        )

    return _impl


cmd_logs = _stub("logs")
cmd_cancel = _stub("cancel")
cmd_cleanup = _stub("cleanup")


def cmd_rescue(cfg: OrchestratorConfig) -> None:
    """Manually trigger Phase D.5 rescue for one or more datasets.

    Useful when the orchestrator's automatic D->D.5 hand-off was
    skipped (e.g. the orchestrator died between Phase D and the
    rescue check, or the operator wants to re-attempt rescue with
    different `--max-rescue-*` settings).

    PLAN.md:374: ``gke/orchestrator.py rescue --run-id ID --dataset NAME``.
    Without ``--dataset``, runs rescue for every dataset in the run.
    """
    if not cfg.explicit_run_id:
        sys.exit(
            "Error: rescue requires --run-id <id> to identify the run "
            "whose state we should consult."
        )

    store = _state.StateStore(cfg.state_uri)
    state = store.read()
    if state is None:
        sys.exit(f"Error: no state at {cfg.state_uri}; cannot rescue.")

    deploy_cfg = _build_deploy_config(cfg)

    failures = []
    for ds in cfg.datasets:
        print()
        print(f"== Rescue {ds} ==")
        try:
            _rescue.phase_d5_rescue_dataset(
                deploy_cfg,
                state,
                store,
                ds,
                src_yaml=cfg.config_yaml,
                base_name=cfg.base_name,
            )
        except _rescue.RescueError as exc:
            failures.append((ds, str(exc)))
            print(f"  Error: {exc}")
        except ValueError as exc:
            failures.append((ds, str(exc)))
            print(f"  Skipped: {exc}")

    if failures:
        print()
        print("Rescue summary — failures:")
        for ds, msg in failures:
            print(f"  {ds}: {msg}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------


def _add_common_flags(p: argparse.ArgumentParser, *, run_flags: bool) -> None:
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML (e.g. gke/regen-gemma3-27b.yaml).",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names. Defaults to the YAML's DATASETS env.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="GCS output directory (gs://...). Defaults to mapping the YAML's "
        "OUTPUT_DIR /gcs mount to gs://.",
    )
    p.add_argument(
        "--project",
        default="cloud-llm-test",
        help="GCP project ID (default: cloud-llm-test).",
    )
    p.add_argument(
        "--gpu-types",
        default="nvidia-h200-141gb,nvidia-h100-80gb",
        help="Comma-separated GPU types to try in Phase C "
        "(default: nvidia-h200-141gb,nvidia-h100-80gb).",
    )
    p.add_argument(
        "--clusters",
        default=None,
        help="Comma-separated cluster:region pairs (default: auto-discover).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-dataset schedule timeout in seconds (default: 3600).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Resume an existing run by id. Without --run-id, `run` always "
        "starts a fresh run (auto-discovery is deferred).",
    )
    p.add_argument(
        "--state-uri",
        default=None,
        help="Override the auto-derived gs://.../<run_id>/state.json path.",
    )
    if run_flags:
        p.add_argument(
            "--new",
            action="store_true",
            help="Force a fresh run_id (mutually exclusive with --run-id).",
        )
        p.add_argument(
            "--from",
            dest="from_phase",
            choices=PHASE_ORDER,
            default=None,
            help="Skip phases earlier than this one. State is still consulted "
            "for the phases that do run.",
        )
        p.add_argument(
            "--ignore-config-change",
            action="store_true",
            help="Proceed even if the YAML's sha256 doesn't match the run's "
            "recorded config_hash.",
        )
        p.add_argument(
            "--force-rebuild",
            action="store_true",
            help="Bypass the GAR-cached image check in Phase B and always "
            "build/push a fresh tag (suffixed with a wall-clock salt).",
        )
        p.add_argument(
            "--enable-scheduling-v2",
            action="store_true",
            help="No-op as of step 6 (the orchestrator always uses strict "
            "scheduling). Kept for back-compat; only `gke/deploy.py "
            "--execute` (legacy racing pattern) still consults the flag.",
        )
        # Step 6: per-(cluster, gpu) candidate budget.
        p.add_argument(
            "--attempt-deadline",
            type=int,
            default=600,
            dest="attempt_deadline",
            help="Per-candidate scheduling deadline in seconds "
            "(default: 600). When the deadline expires and pods aren't "
            "fully scheduled, fall over to the next (cluster, gpu).",
        )
        p.add_argument(
            "--monitor-deadline",
            type=int,
            default=14400,
            dest="monitor_deadline",
            help="How long Phase D will wait for a successfully scheduled "
            "job to finish before declaring it stuck (default: 14400 = 4h). "
            "Used by step 7's rescue trigger.",
        )
        p.add_argument(
            "--spot-exhausted-strategy",
            choices=("sleep_retry", "raise"),
            default="sleep_retry",
            dest="spot_exhausted_strategy",
            help="When every candidate is CAPACITY_EXHAUSTED: 'sleep_retry' "
            "waits --spot-retry-interval and re-runs the candidate loop "
            "until --max-wait-hours; 'raise' bails immediately.",
        )
        p.add_argument(
            "--spot-retry-interval",
            type=int,
            default=900,
            dest="spot_retry_interval",
            help="Seconds to sleep between spot-retry rounds (default: 900).",
        )
        p.add_argument(
            "--max-wait-hours",
            type=int,
            default=12,
            dest="max_wait_hours",
            help="Cap on total spot-retry wait before giving up (default: 12).",
        )
        # Step 7 — rescue (Phase D.5).
        p.add_argument(
            "--max-rescue-attempts",
            type=int,
            default=3,
            dest="max_rescue_attempts",
            help="How many rescue rounds to run before giving up on a "
            "dataset (default: 3).",
        )
        p.add_argument(
            "--max-rescue-shards",
            type=int,
            default=4,
            dest="max_rescue_shards",
            help="Cap on the rescue job's pod count (default: 4). Rescue "
            "jobs only process the missing chunks; this prevents "
            "accidentally launching N=64 pods to process 4 missing chunks.",
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gke/orchestrator.py",
        description="SpecForge GKE regen orchestrator (see gke/PLAN.md).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser(
        "run",
        help="Run the full pipeline (Phases A/B/E native; C/D via gke.lib).",
    )
    _add_common_flags(p_run, run_flags=True)

    p_status = sub.add_parser("status", help="Print state summary for a run.")
    _add_common_flags(p_status, run_flags=False)

    p_rescue = sub.add_parser(
        "rescue",
        help="Manually trigger Phase D.5 rescue for an existing run.",
    )
    _add_common_flags(p_rescue, run_flags=True)

    for name in ("logs", "cancel", "cleanup"):
        p_stub = sub.add_parser(
            name, help=f"(stub) lands in a later step of gke/PLAN.md"
        )
        _add_common_flags(p_stub, run_flags=False)

    args = parser.parse_args(argv)
    cfg = OrchestratorConfig.from_args(args)

    dispatch = {
        "run": cmd_run,
        "status": cmd_status,
        "logs": cmd_logs,
        "cancel": cmd_cancel,
        "cleanup": cmd_cleanup,
        "rescue": cmd_rescue,
    }
    dispatch[args.cmd](cfg)


if __name__ == "__main__":
    main()
