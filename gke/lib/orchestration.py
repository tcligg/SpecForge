"""Per-dataset deploy orchestration.

Houses two algorithms:

* ``deploy_dataset`` — the legacy candidate-racing pattern preserved
  for ``gke/deploy.py --execute``. Submits to every (cluster x gpu)
  combo at once and adopts the first to schedule ``min_running``
  pods. Subject to the silent partial-completion data-loss problem
  PLAN.md:21-24 calls out.

* ``deploy_dataset_strict`` — step-6 replacement. Iterates candidates
  serially with a per-candidate ``attempt_deadline``, requires *full*
  parallelism (``min_running = total_pods``), and uses
  ``gke.scheduling.diagnose_job`` to decide whether to keep waiting,
  fall over to the next candidate, or bail the dataset entirely. On
  global capacity exhaustion (every candidate failed), sleeps for
  ``spot_retry_interval`` and retries until ``max_wait_hours``.

Step 8 will delete ``deploy_dataset`` and the deprecation shim it
serves. The two functions live in the same module to make that a
one-file deletion.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

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

    # Step 5: lazy import so the classifier (and its dependency on
    # subprocess+kubectl JSON parsing) isn't loaded unless the flag
    # is actually set. Keeps this module's import-time footprint
    # unchanged for callers that aren't on scheduling-v2 yet.
    diagnose_fn = None
    if cfg.enable_scheduling_v2:
        from gke.scheduling import diagnose_job, format_summary

        diagnose_fn = (diagnose_job, format_summary)

    while elapsed < cfg.schedule_timeout:
        for cluster, region, gpu in submitted:
            job_name = make_job_name(cfg.base_name, dataset, gpu, region)

            if not use_cluster(cluster, region, cfg.project):
                continue

            running, pending, other = get_pod_status(job_name)

            if running >= cfg.min_running:
                winner = (cluster, region, gpu)
                break

            if diagnose_fn is not None and pending > 0:
                # Observation only in step 5: we log the classifier's
                # view of why pods are pending, but the wait loop
                # above still uses the legacy min_running threshold.
                # Step 6 turns these classifications into scheduling
                # decisions.
                diagnose_job, format_summary = diagnose_fn
                diagnoses = diagnose_job(job_name)
                summary = format_summary(diagnoses)
                print(f"    [v2] {job_name}: {summary}")

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


# ---------------------------------------------------------------------------
# Step 6 — strict-scheduling deploy
# ---------------------------------------------------------------------------


class CandidateResult(enum.Enum):
    """Per-candidate outcome from ``_attempt_candidate``."""

    WIN = "win"  # job fully scheduled within deadline
    TIMEOUT = "timeout"  # deadline passed with pods still pending
    CAPACITY_EXHAUSTED = "capacity_exhausted"  # classifier says no capacity
    CONFIG_ERROR = "config_error"  # operator-fixable; bail dataset
    IMAGE_PULL_ERROR = "image_pull_error"  # operator-fixable; bail dataset
    SUBMIT_FAILED = "submit_failed"  # kubectl apply failed


@dataclass
class StrictDeployResult:
    """Outcome of ``deploy_dataset_strict``.

    Either ``winner`` is set (job fully scheduled, ready for monitor)
    or one of the failure fields explains why we gave up. Step 7's
    rescue path keys off ``failed_candidates`` to know which
    cluster/GPU combinations are off-limits for the rescue job.
    """

    dataset: str
    winner: Optional[Tuple[str, str, str]] = None  # (job_name, cluster, region)
    winner_gpu: str = ""
    bail_reason: Optional[str] = None  # set on CONFIG_ERROR / IMAGE_PULL_ERROR
    spot_exhausted: bool = False  # all candidates exhausted, sleep_retry hit cap
    failed_candidates: List[Tuple[str, str, str, CandidateResult]] = dataclasses.field(
        default_factory=list
    )

    @property
    def succeeded(self) -> bool:
        return self.winner is not None


class StrictDeployError(RuntimeError):
    """Raised on operator-fixable failures (CONFIG_ERROR / IMAGE_PULL_ERROR)
    so the orchestrator can short-circuit the whole pipeline rather than
    silently move on to the next dataset."""


def _candidate_iter(cfg: Config) -> Iterable[Tuple[str, str, str]]:
    """Order candidates as (cluster, region, gpu) tuples.

    PLAN.md "open items deferred" notes per-cluster historical
    success-rate ordering as a future improvement; for v1 we use
    declaration order: outer loop GPU types (preferred GPU first),
    inner loop clusters. This matches operator intent that
    ``--gpu-types nvidia-h200-141gb,nvidia-h100-80gb`` means "try
    H200 first across all clusters before falling back to H100."
    """
    for gpu in cfg.gpu_types:
        for cluster, region in cfg.clusters:
            yield cluster, region, gpu


def _attempt_candidate(
    cfg: Config,
    dataset: str,
    cluster: str,
    region: str,
    gpu: str,
) -> Tuple[CandidateResult, Optional[str]]:
    """Submit + wait for one candidate; classify the outcome.

    Returns ``(result, job_name_or_None)``. The job_name is set when
    submission succeeded so the caller can delete on TIMEOUT /
    CAPACITY_EXHAUSTED before moving on.
    """
    job_name = make_job_name(cfg.base_name, dataset, gpu, region)
    print(f"\n  Attempting {job_name} ({gpu_short_name(gpu)} on {cluster})")

    if not use_cluster(cluster, region, cfg.project):
        print(f"  {cluster}: failed to connect kubectl, skipping.")
        return CandidateResult.SUBMIT_FAILED, None

    success, err = patch_and_apply_yaml(cfg, dataset, gpu, region)
    if not success:
        print(f"  {job_name}: kubectl apply failed.")
        for line in err.splitlines()[:5]:
            print(f"    {line}")
        return CandidateResult.SUBMIT_FAILED, None

    print(
        f"  {job_name}: submitted; waiting up to {cfg.attempt_deadline}s "
        f"for full scheduling ({cfg.total_pods} pods)"
    )

    # Lazy import: keeps gke.scheduling out of the legacy import path.
    from gke.scheduling import (
        PendingReason,
        diagnose_job,
        format_summary,
    )

    elapsed = 0
    poll_interval = 30  # seconds between checks
    while elapsed < cfg.attempt_deadline:
        running, pending, other = get_pod_status(job_name)

        # Strict win condition: every shard must be Running. Anything
        # less risks the silent-partial-completion data-loss path
        # PLAN.md:21-24 was written to fix.
        if running >= cfg.total_pods:
            print(f"  {job_name}: WIN — {running}/{cfg.total_pods} pods running.")
            return CandidateResult.WIN, job_name

        # Classifier-driven branching. ``diagnose_job`` shells out to
        # kubectl; cheap when called every 30s.
        diagnoses = diagnose_job(job_name)
        summary = format_summary(diagnoses)
        print(
            f"    [{elapsed:>4}s] {job_name}: "
            f"running={running} pending={pending} other={other} | {summary}"
        )

        # Operator-fixable failures: bail immediately. Trying the next
        # candidate would just hit the same error.
        for d in diagnoses:
            if d.reason == PendingReason.CONFIG_ERROR:
                print(f"  {job_name}: CONFIG_ERROR on {d.pod_name} -- {d.raw_message}")
                return CandidateResult.CONFIG_ERROR, job_name
            if d.reason == PendingReason.IMAGE_PULL_ERROR:
                print(
                    f"  {job_name}: IMAGE_PULL_ERROR on {d.pod_name} -- {d.raw_message}"
                )
                return CandidateResult.IMAGE_PULL_ERROR, job_name

        # Capacity decision: if every pending pod is CAPACITY_EXHAUSTED
        # (no recent scale-up event), there's no point waiting out the
        # full deadline — fail over now. CAPACITY_WAIT or UNKNOWN keep
        # us waiting.
        pending_diagnoses = [d for d in diagnoses if d.reason is not None]
        if pending_diagnoses and all(
            d.reason == PendingReason.CAPACITY_EXHAUSTED for d in pending_diagnoses
        ):
            print(
                f"  {job_name}: CAPACITY_EXHAUSTED for all {len(pending_diagnoses)} "
                f"pending pods; falling over to next candidate."
            )
            return CandidateResult.CAPACITY_EXHAUSTED, job_name

        time.sleep(poll_interval)
        elapsed += poll_interval

    print(f"  {job_name}: TIMEOUT after {cfg.attempt_deadline}s.")
    return CandidateResult.TIMEOUT, job_name


def _delete_quiet(job_name: str) -> None:
    """Best-effort delete of a losing candidate; never raise."""
    try:
        delete_job(job_name)
        print(f"  {job_name}: deleted.")
    except Exception as exc:  # pragma: no cover - delete is non-fatal
        print(f"  {job_name}: delete attempt failed (non-fatal): {exc}")


def deploy_dataset_strict(cfg: Config, dataset: str) -> StrictDeployResult:
    """Strict-scheduling deploy for one dataset.

    Iterates ``_candidate_iter(cfg)`` serially. For each (cluster, gpu):

    1. Submit. If kubectl apply fails, record and try next.
    2. Wait up to ``cfg.attempt_deadline`` seconds for *all*
       ``cfg.total_pods`` shards to be Running.
    3. Use ``gke.scheduling.diagnose_job`` to classify pending pods:
       * CONFIG_ERROR / IMAGE_PULL_ERROR -> raise StrictDeployError.
       * CAPACITY_EXHAUSTED for all pending -> delete + next.
       * CAPACITY_WAIT / UNKNOWN -> keep waiting until deadline.
    4. WIN -> record state, return.

    If every candidate fails:
    * ``cfg.spot_exhausted_strategy == "sleep_retry"``: sleep
      ``cfg.spot_retry_interval`` and recurse, capped by
      ``cfg.max_wait_hours``.
    * Else: return with ``spot_exhausted=True``.
    """
    print()
    print("=" * 60)
    print(f"  Deploying (strict): {dataset}")
    print(f"  GPU types: {', '.join(gpu_short_name(g) for g in cfg.gpu_types)}")
    print(f"  Clusters:  {len(cfg.clusters)}")
    print(f"  Target:    {cfg.total_pods} pods (strict)")
    print(f"  Per-attempt deadline: {cfg.attempt_deadline}s")
    print("=" * 60)

    deadline_wall = time.monotonic() + cfg.max_wait_hours * 3600
    result = StrictDeployResult(dataset=dataset)
    rescue_round = 0

    while True:
        rescue_round += 1
        if rescue_round > 1:
            print(
                f"\n  [round {rescue_round}] retrying candidates after "
                f"spot-exhausted sleep."
            )

        for cluster, region, gpu in _candidate_iter(cfg):
            outcome, job_name = _attempt_candidate(cfg, dataset, cluster, region, gpu)

            if outcome == CandidateResult.WIN:
                assert job_name is not None
                result.winner = (job_name, cluster, region)
                result.winner_gpu = gpu
                return result

            if outcome in (
                CandidateResult.CONFIG_ERROR,
                CandidateResult.IMAGE_PULL_ERROR,
            ):
                # Operator-fixable: delete the broken job and bail
                # the whole dataset. Trying the next candidate would
                # hit the same error (image is global; config errors
                # are usually YAML-level).
                if job_name:
                    _delete_quiet(job_name)
                result.failed_candidates.append((cluster, region, gpu, outcome))
                result.bail_reason = outcome.value
                raise StrictDeployError(
                    f"{dataset}: {outcome.value} on {job_name or '<not submitted>'} "
                    f"({cluster}, {gpu_short_name(gpu)}). Operator action required."
                )

            # CAPACITY_EXHAUSTED, TIMEOUT, SUBMIT_FAILED: fall over.
            if job_name:
                _delete_quiet(job_name)
            result.failed_candidates.append((cluster, region, gpu, outcome))

        # Every candidate exhausted in this round.
        print(
            f"\n  All {len(result.failed_candidates)} candidates failed in round "
            f"{rescue_round}. Strategy: {cfg.spot_exhausted_strategy}"
        )
        if cfg.spot_exhausted_strategy != "sleep_retry":
            result.spot_exhausted = True
            return result

        remaining = deadline_wall - time.monotonic()
        if remaining <= 0:
            print(f"  --max-wait-hours={cfg.max_wait_hours} reached; giving up.")
            result.spot_exhausted = True
            return result

        sleep_for = min(cfg.spot_retry_interval, max(int(remaining), 1))
        print(
            f"  Sleeping {sleep_for}s before next round (remaining "
            f"{int(remaining)}s of {cfg.max_wait_hours * 3600}s budget)."
        )
        time.sleep(sleep_for)
        # Clear failed-candidate list for the next round so the
        # caller's record reflects only the final attempt set.
        result.failed_candidates = []
