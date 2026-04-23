"""Phase D.5 — rescue path for partially-completed regen jobs.

Step 7 of gke/PLAN.md. The primary job (Phases C/D) sometimes fails
to process every chunk: spot preemption mid-shard, scheduling
exhaustion that leaves some shards Pending past the deadline,
backoffLimitPerIndex exhaustion on a small subset, etc. Rather than
re-running the whole dataset (expensive — most chunks are already
done) Phase D.5 inspects the chunk markers in GCS, computes which
chunks are still missing, and submits a smaller job that only
processes those.

Trigger conditions (PLAN.md:357-360, evaluated by the orchestrator):

* ``state.monitor[ds].primary_job_status == "Failed"``
* ``state.monitor[ds].primary_job_status == "Active"`` past
  ``monitor_deadline`` (orchestrator wall-clocks this; we just see
  the Active state when the orchestrator decides to invoke us)
* Mixed shard status (some chunks done, some missing) — covered by
  ``compute_missing_chunks`` returning a non-empty list

Algorithm (PLAN.md:362-372):

    for attempt in range(1, max_rescue_attempts + 1):
        missing = compute_missing_chunks(state, ds)
        if not missing: mark all_chunks_done; return
        rescue_shards = min(len(missing), max_rescue_shards)
        submit_rescue_job(ds, attempt, rescue_shards, missing, candidates)
        monitor_result = monitor_job(...)
        append_rescue_attempt(...)
    final missing != 0 -> RescueError

The rescue submission re-uses ``deploy_dataset_strict`` for the
candidate-iteration / strict-scheduling logic. The only new bits
are (a) computing the missing-chunk list from GCS markers and
(b) patching the YAML so the rescue job has fewer shards and
forwards ``--chunk-ids`` to the regen script (via
``EXTRA_REGEN_ARGS`` env, which the entrypoint already shlex-splits
onto the regen command line).
"""

from __future__ import annotations

import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from gke import state as _state
from gke.lib import merge as _merge
from gke.lib.config import Config
from gke.lib.k8s import wait_for_job_completion
from gke.lib.orchestration import (
    StrictDeployError,
    StrictDeployResult,
    deploy_dataset_strict,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RescueError(RuntimeError):
    """Raised when Phase D.5 exhausts all attempts and chunks remain."""


# ---------------------------------------------------------------------------
# Missing-chunks math
# ---------------------------------------------------------------------------


_DONE_RE = re.compile(r".*/chunk_(\d{6})\.done$")


def _list_done_markers(chunks_dir: str) -> List[int]:
    """Return the chunk IDs whose ``.done`` markers exist in GCS.

    Empty list on failure or when the chunks directory is missing
    (treated as "all chunks missing"; caller will then submit the
    full set, which is the right thing if the primary job died
    early).
    """
    proc = subprocess.run(
        ["gsutil", "ls", f"{chunks_dir}/chunk_*.done"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    ids: List[int] = []
    for line in proc.stdout.splitlines():
        m = _DONE_RE.match(line.strip())
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def compute_missing_chunks(
    state: _state.RunState,
    deploy_cfg: Config,
    dataset: str,
) -> List[int]:
    """Compute the list of chunk IDs that have no ``.done`` marker yet.

    Reads:
    * ``state.phases.prepare[dataset].n_lines`` for the input size.
    * ``deploy_cfg.chunk_size`` for the chunk slicing.
    * ``gke.lib.merge.chunks_dir_uri`` for where the chunks live in
      GCS.

    Returns a sorted list of missing chunk IDs. An empty list means
    "every chunk has a .done marker; nothing to rescue."

    Raises ``ValueError`` if state lacks the input-size record (Phase A
    didn't run for this dataset) or if chunk_size is zero — both are
    operator-fixable rather than transient.
    """
    prep = state.phases.prepare.get(dataset) or {}
    n_lines = int(prep.get("n_lines") or 0)
    if n_lines <= 0:
        raise ValueError(
            f"rescue needs state.prepare[{dataset!r}].n_lines; "
            f"Phase A may not have completed for this dataset."
        )
    chunk_size = int(deploy_cfg.chunk_size or 0)
    if chunk_size <= 0:
        raise ValueError(f"rescue needs chunk_size > 0; got {chunk_size!r}")

    total_chunks = math.ceil(n_lines / chunk_size)

    chunks_dir = _merge.chunks_dir_uri(deploy_cfg, dataset)
    if chunks_dir is None:
        # Same as "no chunks done": every chunk is missing.
        return list(range(total_chunks))

    done = set(_list_done_markers(chunks_dir))
    missing = [cid for cid in range(total_chunks) if cid not in done]
    return missing


# ---------------------------------------------------------------------------
# YAML patching for rescue jobs
# ---------------------------------------------------------------------------


def _write_rescue_yaml(
    *,
    src_yaml: str,
    dst_yaml: str,
    base_name: str,
    rescue_shards: int,
    chunk_ids: Sequence[int],
) -> None:
    """Produce a rescue YAML by patching the primary YAML.

    Three string-level edits, mirroring the existing
    ``patch_and_apply_yaml`` style (PLAN.md "open items" tracks the
    move to ruamel.yaml):

    1. ``metadata.name`` -> ``<base>-rescue`` so deploy_dataset_strict's
       per-(cluster, gpu) suffixing produces unique job names that
       don't collide with the primary job.
    2. ``completions: N`` and ``parallelism: N`` -> ``rescue_shards``.
    3. Append ``EXTRA_REGEN_ARGS=\"--chunk-ids 12,17,...\"`` to the
       env list. We surgically inject before the existing
       ``volumeMounts:`` line so YAML structure stays valid; if there
       is already an ``EXTRA_REGEN_ARGS`` entry we replace its value.
    """
    with open(src_yaml) as f:
        text = f.read()

    # Patch metadata name.
    text = re.sub(
        r"^(\s*name:\s*)" + re.escape(base_name) + r"\s*$",
        rf"\1{base_name}-rescue",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # Patch completions and parallelism.
    text = re.sub(
        r"^(\s*completions:\s*)\d+\s*$",
        rf"\g<1>{rescue_shards}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(\s*parallelism:\s*)\d+\s*$",
        rf"\g<1>{rescue_shards}",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # Set EXTRA_REGEN_ARGS. We support both shapes:
    #   - replace existing ``- name: EXTRA_REGEN_ARGS / value: "..."``
    #   - inject a new one before the closing ``volumeMounts:`` line
    chunk_ids_csv = ",".join(str(c) for c in chunk_ids)
    new_value = f"--chunk-ids {chunk_ids_csv}"

    extra_block_re = re.compile(
        r"(-\s*name:\s*EXTRA_REGEN_ARGS\s*[\r\n]+\s*value:\s*\"?)[^\"\r\n]*(\"?)",
    )
    if extra_block_re.search(text):
        text = extra_block_re.sub(
            lambda m: f"{m.group(1)}{new_value}{m.group(2) or ''}",
            text,
            count=1,
        )
    else:
        # Inject before the first ``volumeMounts:`` line at the right
        # indent. The regex captures the indent so we mirror it on the
        # new env entries.
        m = re.search(r"^(\s*)volumeMounts:\s*$", text, re.MULTILINE)
        if m is None:
            raise RuntimeError(
                f"rescue: could not find volumeMounts: anchor in {src_yaml}; "
                "cannot inject EXTRA_REGEN_ARGS."
            )
        indent = m.group(1)
        injection = (
            f'{indent}- name: EXTRA_REGEN_ARGS\n{indent}  value: "{new_value}"\n'
        )
        # Insert before the volumeMounts line.
        text = text[: m.start()] + injection + text[m.start() :]

    with open(dst_yaml, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Rescue orchestration
# ---------------------------------------------------------------------------


@dataclass
class RescueAttemptResult:
    """One row in ``state.phases.rescue[dataset].attempts``."""

    attempt: int
    cluster: str
    region: str
    gpu: str
    job_name: str
    covered_chunks: List[int]
    total_shards: int
    completed_at: str
    result: str  # "Complete" | "Failed" | "NoCandidates"

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "cluster": self.cluster,
            "region": self.region,
            "gpu": self.gpu,
            "job_name": self.job_name,
            "covered_chunks": list(self.covered_chunks),
            "total_shards": self.total_shards,
            "completed_at": self.completed_at,
            "result": self.result,
        }


def _record_attempt(
    store: _state.StateStore,
    dataset: str,
    attempt: RescueAttemptResult,
) -> None:
    def _mut(s: _state.RunState) -> _state.RunState:
        rec = s.phases.rescue.get(dataset) or {"attempts": [], "all_chunks_done": False}
        rec.setdefault("attempts", []).append(attempt.to_dict())
        s.phases.rescue[dataset] = rec
        return s

    store.update(_mut)


def _record_done(store: _state.StateStore, dataset: str) -> None:
    def _mut(s: _state.RunState) -> _state.RunState:
        rec = s.phases.rescue.get(dataset) or {"attempts": [], "all_chunks_done": False}
        rec["all_chunks_done"] = True
        s.phases.rescue[dataset] = rec
        return s

    store.update(_mut)


def phase_d5_rescue_dataset(
    deploy_cfg: Config,
    state: _state.RunState,
    store: _state.StateStore,
    dataset: str,
    *,
    src_yaml: str,
    base_name: str,
) -> None:
    """Run rescue rounds until missing == 0 or attempts exhausted.

    Each attempt:
      1. ``compute_missing_chunks(state, deploy_cfg, dataset)``.
      2. Patch the YAML into ``<src_yaml>.rescue.attempt<N>.yaml``
         with ``completions = min(len(missing), max_rescue_shards)``
         and ``EXTRA_REGEN_ARGS=--chunk-ids ...``.
      3. Substitute that YAML into a copy of ``deploy_cfg`` and call
         ``deploy_dataset_strict`` to acquire a candidate.
      4. ``wait_for_job_completion`` for the rescue job.
      5. Record the attempt in state.

    On success (missing == 0 after some attempt), records
    ``all_chunks_done=True`` and returns. On exhaustion, raises
    ``RescueError`` with the still-missing chunk count so the caller
    can surface a useful message.
    """
    print()
    print("=" * 60)
    print(f"  Phase D.5 — Rescue: {dataset}")
    print(f"  Max attempts: {deploy_cfg.max_rescue_attempts}")
    print(f"  Max shards/attempt: {deploy_cfg.max_rescue_shards}")
    print("=" * 60)

    for attempt in range(1, deploy_cfg.max_rescue_attempts + 1):
        missing = compute_missing_chunks(state, deploy_cfg, dataset)
        if not missing:
            print(
                f"  [done] {dataset}: all chunks present after attempt {attempt - 1}."
            )
            _record_done(store, dataset)
            return

        rescue_shards = min(len(missing), deploy_cfg.max_rescue_shards)
        print(
            f"\n  [attempt {attempt}/{deploy_cfg.max_rescue_attempts}] "
            f"{len(missing)} missing chunks; submitting {rescue_shards}-shard rescue."
        )

        rescue_yaml = f"{src_yaml}.rescue.attempt{attempt}.yaml"
        rescue_base = f"{base_name}-rescue"
        _write_rescue_yaml(
            src_yaml=src_yaml,
            dst_yaml=rescue_yaml,
            base_name=base_name,
            rescue_shards=rescue_shards,
            chunk_ids=missing,
        )

        # Substitute into a copy of deploy_cfg so the strict-scheduling
        # path uses the rescue YAML and the rescue base name. We keep
        # the candidate list (clusters x GPUs) as configured; rescue
        # jobs are usually small enough to fit on whatever spot
        # capacity is available.
        rescue_cfg = _clone_config(deploy_cfg)
        rescue_cfg.job_yaml = rescue_yaml
        rescue_cfg.base_name = rescue_base
        rescue_cfg.total_pods = rescue_shards
        rescue_cfg.min_running = rescue_shards

        # Re-read state for fresh values.
        state = store.read() or state

        try:
            result: StrictDeployResult = deploy_dataset_strict(rescue_cfg, dataset)
        except StrictDeployError as exc:
            attempt_rec = RescueAttemptResult(
                attempt=attempt,
                cluster="",
                region="",
                gpu="",
                job_name="",
                covered_chunks=missing,
                total_shards=rescue_shards,
                completed_at=_state._utc_now_iso(),
                result=f"DeployError: {exc}",
            )
            _record_attempt(store, dataset, attempt_rec)
            raise RescueError(
                f"{dataset}: rescue attempt {attempt} hit operator-fixable error: {exc}"
            ) from exc

        if not result.succeeded:
            attempt_rec = RescueAttemptResult(
                attempt=attempt,
                cluster="",
                region="",
                gpu="",
                job_name="",
                covered_chunks=missing,
                total_shards=rescue_shards,
                completed_at=_state._utc_now_iso(),
                result="NoCandidates",
            )
            _record_attempt(store, dataset, attempt_rec)
            print(
                f"  [attempt {attempt}] no candidate scheduled the rescue job; "
                f"{len(missing)} chunks still missing."
            )
            continue

        job_name, cluster, region = result.winner
        print(f"  [attempt {attempt}] rescue scheduled: {job_name}; monitoring...")
        ok = wait_for_job_completion(
            job_name, cluster, region, rescue_cfg, timeout=rescue_cfg.monitor_deadline
        )

        attempt_rec = RescueAttemptResult(
            attempt=attempt,
            cluster=cluster,
            region=region,
            gpu=result.winner_gpu,
            job_name=job_name,
            covered_chunks=missing,
            total_shards=rescue_shards,
            completed_at=_state._utc_now_iso(),
            result="Complete" if ok else "Failed",
        )
        _record_attempt(store, dataset, attempt_rec)

        if not ok:
            print(f"  [attempt {attempt}] rescue job {job_name} failed/timed out.")
            # Loop continues; next attempt re-computes missing chunks
            # (some may have succeeded even on a "failed" job).
            time.sleep(5)
            state = store.read() or state
            continue

        # Re-load state and re-check missing on the next iteration.
        state = store.read() or state

    # Exhausted attempts.
    final_missing = compute_missing_chunks(state, deploy_cfg, dataset)
    if not final_missing:
        _record_done(store, dataset)
        return
    raise RescueError(
        f"{dataset}: {len(final_missing)} chunks still missing after "
        f"{deploy_cfg.max_rescue_attempts} attempts."
    )


def _clone_config(cfg: Config) -> Config:
    """Shallow-copy a Config dataclass so we can swap fields without
    mutating the caller's instance."""
    new = Config()
    for fname in cfg.__dataclass_fields__:
        setattr(new, fname, getattr(cfg, fname))
    return new
