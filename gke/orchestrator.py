#!/usr/bin/env python3
"""Single-entrypoint orchestrator for the SpecForge GKE regen pipeline.

Step 4a of gke/PLAN.md. This module is the future home of the full
six-phase pipeline:

    Phase A — PREPARE   (host)   prepare_data.py per dataset -> GCS
    Phase B — BUILD     (host)   build_and_push.sh, tag = git-sha
    Phase C — DEPLOY    (GKE)    submit job; wait for FULL parallelism
    Phase D — MONITOR   (GKE)    poll until terminal; per-shard status
    Phase D.5 — RESCUE  (GKE)    if partial: dispatch shard-aware rescue
    Phase E — MERGE     (host)   recursive gsutil compose to single jsonl

In step 4a only Phase A is implemented natively. Phases B/C/D/E delegate
to ``gke/deploy.py --execute`` so end-to-end runs keep working while we
move logic in piecewise. The delegated path inherits the step-3 state
observer when ``--state-uri`` is wired through, so resume metadata is
populated even before each phase becomes native.

CLI shape::

    python3 gke/orchestrator.py run --config gke/regen-gemma3-27b.yaml \
        [--datasets a,b]               # override YAML's DATASETS env
        [--output-dir gs://bucket/p]   # override YAML's OUTPUT_DIR
        [--run-id ID]                  # resume an existing run
        [--new]                        # force a fresh run_id
        [--from prepare|build|deploy|monitor|merge]
        [--state-uri gs://...]         # override derived state path
        [--ignore-config-change]       # accept config_hash mismatch
        [--force-rebuild]              # ignore cached build (step 4b)

    python3 gke/orchestrator.py status --config <yaml> [--run-id ID]
    python3 gke/orchestrator.py logs    ...   # stub
    python3 gke/orchestrator.py cancel  ...   # stub
    python3 gke/orchestrator.py cleanup ...   # stub
    python3 gke/orchestrator.py rescue  ...   # stub

Resume rules locked in for step 4a:

* ``run`` without ``--run-id`` always starts a new run. Resuming requires
  passing ``--run-id <id>`` explicitly. Auto-discovery by ``config_hash``
  is deferred until later steps.
* The state URI defaults to
  ``gs://<bucket>/<output-prefix>/<run_id>/state.json`` derived from the
  YAML's ``OUTPUT_DIR`` env (``/gcs/<bucket>/<path>``). ``--state-uri``
  overrides.

Only the flags listed in PLAN.md as "functional in this step" actually
do anything in 4a: ``--run-id``, ``--new``, ``--from``. The others are
parsed and stored on the config so step 4b/5/6 can light them up
without re-touching argparse.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Make sure ``gke`` is importable when this file is invoked as a script
# from the repo root (``python3 gke/orchestrator.py ...``).
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gke import state as _state  # noqa: E402

PHASE_ORDER: Tuple[str, ...] = ("prepare", "build", "deploy", "monitor", "merge")


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

    # Run identity ------------------------------------------------------
    run_id: str
    state_uri: str

    # Resume / behaviour flags -----------------------------------------
    explicit_run_id: bool = False  # True if user passed --run-id (vs --new)
    from_phase: Optional[str] = None
    ignore_config_change: bool = False
    force_rebuild: bool = False

    # Derived from YAML (cached) ---------------------------------------
    base_name: str = ""
    yaml_output_dir: Optional[str] = None  # raw OUTPUT_DIR env (for reference)
    yaml_prepare_dir: Optional[str] = None  # PREPARE_DATA_DIR env

    # Pass-through to the delegated deploy CLI -------------------------
    delegate_extra_args: List[str] = field(default_factory=list)

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

        return cls(
            config_yaml=os.path.abspath(config_yaml),
            datasets=datasets,
            output_dir=output_dir,
            project=args.project,
            run_id=run_id,
            state_uri=state_uri,
            explicit_run_id=explicit,
            from_phase=from_phase,
            ignore_config_change=bool(getattr(args, "ignore_config_change", False)),
            force_rebuild=bool(getattr(args, "force_rebuild", False)),
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
# Phases B/C/D/E — stubs that delegate to gke/deploy.py --execute
# ---------------------------------------------------------------------------


def _delegate_to_deploy_cli(cfg: OrchestratorConfig) -> None:
    """Run ``gke/deploy.py --execute`` to cover B/C/D/E in one shell-out.

    Step 4a punts the per-phase logic. The deploy CLI already knows how
    to build/push, deploy across clusters, monitor, and merge, and since
    step 3 it threads its own state observer when ``--state-uri`` is
    set. We pass the orchestrator's ``state_uri`` through so the same
    state document gets phase-B/C/D/E records appended.
    """
    deploy_script = _PROJECT_ROOT / "gke" / "deploy.py"
    if not deploy_script.is_file():
        sys.exit(f"Error: {deploy_script} not found.")

    cmd = [
        sys.executable,
        str(deploy_script),
        "--yaml",
        cfg.config_yaml,
        "--datasets",
        ",".join(cfg.datasets),
        "--project",
        cfg.project,
        "--state-uri",
        cfg.state_uri,
        "--execute",
    ]
    if cfg.output_dir:
        cmd += ["--output-dir", cfg.output_dir]
    if cfg.delegate_extra_args:
        cmd += list(cfg.delegate_extra_args)

    print()
    print("[Phases B-E] Delegating to gke/deploy.py --execute")
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"Error: deploy.py --execute failed (exit {proc.returncode})")


def phase_b_build(
    cfg: OrchestratorConfig, store: _state.StateStore, state: _state.RunState
) -> None:
    """Stub: covered by ``_delegate_to_deploy_cli``. Native impl in step 4b."""
    return None


def phase_c_deploy(
    cfg: OrchestratorConfig, store: _state.StateStore, state: _state.RunState
) -> None:
    """Stub: covered by ``_delegate_to_deploy_cli``. Native impl in step 6."""
    return None


def phase_d_monitor(
    cfg: OrchestratorConfig, store: _state.StateStore, state: _state.RunState
) -> None:
    """Stub: covered by ``_delegate_to_deploy_cli``. Native impl in step 6."""
    return None


def phase_e_merge(
    cfg: OrchestratorConfig, store: _state.StateStore, state: _state.RunState
) -> None:
    """Stub: covered by ``_delegate_to_deploy_cli``. Native impl in step 4b."""
    return None


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
    if cfg.from_phase:
        print(f"  Resume:    from phase '{cfg.from_phase}'")
    print("=" * 60)

    store, state = _open_or_create_state(cfg)
    phases = _phases_to_run(cfg)

    if "prepare" in phases:
        phase_a_prepare(cfg, store, state)

    # Phases B/C/D/E are still delegated as a unit. If the user asked
    # to start from build/deploy/monitor/merge, we skip Phase A but
    # still hand the rest off to the deploy CLI in one go. Once each
    # phase is native (steps 4b, 6) this single delegate call gets
    # split into per-phase calls.
    if any(p in phases for p in ("build", "deploy", "monitor", "merge")):
        _delegate_to_deploy_cli(cfg)

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
    print("Per-dataset prepare:")
    for ds in cfg.datasets:
        rec = state.phases.prepare.get(ds) or {}
        print(
            f"  {ds:<28} status={rec.get('status', '-'):<6} "
            f"size={rec.get('size', '-')}  uri={rec.get('uri', '-')}"
        )


def _stub(name: str):
    def _impl(cfg: OrchestratorConfig) -> None:
        sys.exit(
            f"`{name}` is a step 4a stub. It will land in a later step of "
            "gke/PLAN.md. Use the underlying tools in the meantime "
            "(`kubectl`, `gke/deploy.py --status`, `gsutil rm`)."
        )

    return _impl


cmd_logs = _stub("logs")
cmd_cancel = _stub("cancel")
cmd_cleanup = _stub("cleanup")
cmd_rescue = _stub("rescue")


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
            help="Ignore any cached image (parsed in 4a, applied in step 4b).",
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gke/orchestrator.py",
        description="SpecForge GKE regen orchestrator (see gke/PLAN.md).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser(
        "run", help="Run the full pipeline (delegates B-E in step 4a)."
    )
    _add_common_flags(p_run, run_flags=True)

    p_status = sub.add_parser("status", help="Print state summary for a run.")
    _add_common_flags(p_status, run_flags=False)

    for name in ("logs", "cancel", "cleanup", "rescue"):
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
