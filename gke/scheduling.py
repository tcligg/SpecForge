"""Pending-pod reason classifier for the GKE regen pipeline.

Step 5 of gke/PLAN.md. Owns the question "why is this pod still
Pending?" so step 6's strict-scheduling loop can decide whether to
keep waiting, fall over to the next candidate, or bail the dataset.

Step 5 itself is **observation only** — wired behind
``--enable-scheduling-v2`` in ``gke/deploy.py`` so an operator can
validate the classifier's output against real GKE conditions before
step 6 acts on it.

Decision matrix (PLAN.md:307-313, reproduced here as the source of
truth in code form):

    CAPACITY_WAIT       Insufficient nvidia.com/gpu AND TriggeredScaleUp
                        event in last 5 min                  -> wait
    CAPACITY_EXHAUSTED  Insufficient nvidia.com/gpu AND no scale-up
                        event AND deadline passed            -> next
                        candidate (deadline applied in step 6)
    CONFIG_ERROR        volume / secret / affinity error     -> bail
    IMAGE_PULL_ERROR    ImagePullBackOff, ErrImagePull       -> bail
    UNKNOWN             anything else                        -> treat as
                                                              CAPACITY_WAIT
                                                              (step 6)

The classifier is a pure function of (pod_status_dict, events_list)
plus a "now" timestamp, so unit tests use captured kubectl JSON
fixtures with no subprocess plumbing.
"""

from __future__ import annotations

import datetime as _dt
import enum
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

# How recent a TriggeredScaleUp event has to be for a pod to count as
# CAPACITY_WAIT (vs. CAPACITY_EXHAUSTED). PLAN.md:309 specifies 5 min;
# kept as a module constant so step 6 can re-tune without touching
# decision logic.
SCALE_UP_FRESH_SECONDS = 5 * 60

_CAPACITY_GPU_RE = re.compile(r"insufficient\s+nvidia\.com/gpu", re.IGNORECASE)
_CAPACITY_NODE_RE = re.compile(
    r"\b(0/\d+\s+nodes\s+are\s+available|no\s+nodes\s+available)\b",
    re.IGNORECASE,
)
_CONFIG_ERROR_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(?:volume|persistentvolumeclaim|pvc)\b", re.IGNORECASE),
    re.compile(r"\bsecret\s+\".*?\"\s+not\s+found", re.IGNORECASE),
    re.compile(r"\b(?:nodeaffinity|nodeselector|node\s+affinity)\b", re.IGNORECASE),
    re.compile(r"\btoleration\b", re.IGNORECASE),
    re.compile(r"\bconfigmap\b", re.IGNORECASE),
)
_IMAGE_PULL_REASONS = frozenset(
    {"ImagePullBackOff", "ErrImagePull", "InvalidImageName"}
)


class PendingReason(enum.Enum):
    """Why the scheduler hasn't bound this pod to a node yet."""

    CAPACITY_WAIT = "capacity_wait"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    CONFIG_ERROR = "config_error"
    IMAGE_PULL_ERROR = "image_pull_error"
    UNKNOWN = "unknown"


@dataclass
class PodDiagnosis:
    """Per-pod classifier output.

    ``reason`` is None when the pod is not Pending (Running,
    Succeeded, Failed, etc.) — included so summarizers can show a
    full job snapshot, not just the troubled pods.
    """

    pod_name: str
    phase: str
    reason: Optional[PendingReason]
    raw_message: str
    last_event_age_seconds: Optional[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse_k8s_timestamp(value: Optional[str]) -> Optional[_dt.datetime]:
    """Parse a Kubernetes ISO-8601 timestamp (``...Z``) into aware UTC.

    Returns None on any parse error. We accept ``...Z`` because that's
    what kubectl emits for ``lastTimestamp`` and ``eventTime``;
    Python's ``fromisoformat`` only learned to parse ``Z`` in 3.11, so
    we fall back to manual handling for older interpreters.
    """
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _seconds_since(when: Optional[_dt.datetime], now: _dt.datetime) -> Optional[int]:
    if when is None:
        return None
    return int((now - when).total_seconds())


def _event_timestamp(event: Dict[str, Any]) -> Optional[_dt.datetime]:
    """Best-available timestamp for a Kubernetes Event.

    Prefers ``eventTime`` (high-precision, set by API in 1.19+),
    then ``lastTimestamp`` (legacy), then ``firstTimestamp``.
    """
    for key in ("eventTime", "lastTimestamp", "firstTimestamp"):
        value = event.get(key)
        ts = _parse_k8s_timestamp(value if isinstance(value, str) else None)
        if ts is not None:
            return ts
    return None


def _scheduling_message(pod: Dict[str, Any]) -> str:
    """Concatenate every ``PodScheduled``/``Unschedulable`` condition msg."""
    status = pod.get("status") or {}
    conditions = status.get("conditions") or []
    parts: List[str] = []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        msg = cond.get("message")
        if isinstance(msg, str) and msg:
            parts.append(msg)
    return " | ".join(parts)


def _container_status_signals(pod: Dict[str, Any]) -> Iterable[str]:
    """Yield ``reason`` strings from container statuses.

    Looks at both ``containerStatuses`` and ``initContainerStatuses``
    for the ``waiting.reason`` field, which is where ImagePullBackOff
    et al. surface.
    """
    status = pod.get("status") or {}
    for key in ("containerStatuses", "initContainerStatuses"):
        statuses = status.get(key) or []
        for cs in statuses:
            if not isinstance(cs, dict):
                continue
            waiting = (cs.get("state") or {}).get("waiting") or {}
            reason = waiting.get("reason")
            if isinstance(reason, str) and reason:
                yield reason


def _has_recent_scaleup(events: Sequence[Dict[str, Any]], now: _dt.datetime) -> bool:
    """True iff any event named ``TriggeredScaleUp`` is fresh.

    "Fresh" = within ``SCALE_UP_FRESH_SECONDS`` of ``now``. Both the
    cluster autoscaler and GKE Autopilot emit this event reason on
    the pod when a scale-up is in flight.
    """
    cutoff = now - _dt.timedelta(seconds=SCALE_UP_FRESH_SECONDS)
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("reason") != "TriggeredScaleUp":
            continue
        ts = _event_timestamp(event)
        if ts is None:
            # Unknown timestamp — treat as fresh; better to wait one
            # extra cycle than to fail-fast incorrectly.
            return True
        if ts >= cutoff:
            return True
    return False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_pod(
    pod: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    *,
    now: Optional[_dt.datetime] = None,
) -> PodDiagnosis:
    """Pure classifier: (pod, events) -> PodDiagnosis.

    Step 5 evaluates the matrix from PLAN.md:307-313. The deadline
    half of CAPACITY_EXHAUSTED is deferred to step 6 — for now we
    return CAPACITY_EXHAUSTED whenever the pod is GPU-blocked and
    has no fresh scale-up event, leaving step 6 to decide whether
    enough time has passed to act on it.
    """
    now = now or _utcnow()

    metadata = pod.get("metadata") or {}
    pod_name = metadata.get("name") or "<unknown>"
    status = pod.get("status") or {}
    phase = status.get("phase") or "Unknown"

    # Most recent event timestamp, for the diagnosis record (separate
    # from the scale-up freshness check).
    last_event_age: Optional[int] = None
    for event in events:
        if not isinstance(event, dict):
            continue
        ts = _event_timestamp(event)
        age = _seconds_since(ts, now)
        if age is None:
            continue
        if last_event_age is None or age < last_event_age:
            last_event_age = age

    if phase != "Pending":
        return PodDiagnosis(
            pod_name=pod_name,
            phase=phase,
            reason=None,
            raw_message="",
            last_event_age_seconds=last_event_age,
        )

    sched_msg = _scheduling_message(pod)
    container_reasons = list(_container_status_signals(pod))

    # 1. Image-pull failures take precedence — CONFIG_ERROR-style
    # patterns can show up alongside an ImagePullBackOff (e.g. for a
    # missing volume that depends on a healthy container), and bailing
    # on the image is the correct response either way.
    if any(r in _IMAGE_PULL_REASONS for r in container_reasons):
        return PodDiagnosis(
            pod_name=pod_name,
            phase=phase,
            reason=PendingReason.IMAGE_PULL_ERROR,
            raw_message=", ".join(container_reasons) or sched_msg,
            last_event_age_seconds=last_event_age,
        )

    # 2. Config errors before capacity. Real Unschedulable messages
    # have the shape ``0/N nodes are available: <reason>`` where the
    # reason is what we actually care about. PVC / secret / affinity /
    # toleration errors are the operator's fault, not capacity, and
    # need to short-circuit before we attribute the pod to capacity
    # exhaustion just because the prefix mentions nodes.
    for pattern in _CONFIG_ERROR_PATTERNS:
        if pattern.search(sched_msg):
            return PodDiagnosis(
                pod_name=pod_name,
                phase=phase,
                reason=PendingReason.CONFIG_ERROR,
                raw_message=sched_msg,
                last_event_age_seconds=last_event_age,
            )

    # 3. Capacity. GPU exhaustion is the dominant scheduling message
    # in our deployment; the generic "0/N nodes are available" with no
    # config-pattern match means the cluster is full of the wrong
    # node shapes (also a capacity problem).
    is_gpu_blocked = bool(_CAPACITY_GPU_RE.search(sched_msg))
    is_node_blocked = bool(_CAPACITY_NODE_RE.search(sched_msg))
    if is_gpu_blocked or is_node_blocked:
        if _has_recent_scaleup(events, now):
            reason = PendingReason.CAPACITY_WAIT
        else:
            # Step 5 returns EXHAUSTED here; step 6 combines this with
            # an attempt deadline to decide whether to fall over.
            reason = PendingReason.CAPACITY_EXHAUSTED
        return PodDiagnosis(
            pod_name=pod_name,
            phase=phase,
            reason=reason,
            raw_message=sched_msg,
            last_event_age_seconds=last_event_age,
        )

    return PodDiagnosis(
        pod_name=pod_name,
        phase=phase,
        reason=PendingReason.UNKNOWN,
        raw_message=sched_msg,
        last_event_age_seconds=last_event_age,
    )


def diagnose_job(job_name: str) -> List[PodDiagnosis]:
    """Pull all pods for ``job_name`` and classify each.

    Pure orchestration: fetches pod list and per-pod events via the
    helpers in ``gke.lib.k8s`` and runs each through ``classify_pod``.
    Returns an empty list if kubectl returns nothing (e.g. the job
    hasn't created pods yet).
    """
    # Local import to avoid a hard dependency on kubectl availability
    # at import time (the classifier itself is pure).
    from gke.lib.k8s import get_pod_events, get_pods_json

    pods = get_pods_json(job_name)
    if not pods:
        return []

    now = _utcnow()
    diagnoses: List[PodDiagnosis] = []
    for pod in pods:
        if not isinstance(pod, dict):
            continue
        pod_name = (pod.get("metadata") or {}).get("name") or ""
        events = get_pod_events(pod_name) if pod_name else []
        diagnoses.append(classify_pod(pod, events, now=now))
    return diagnoses


# ---------------------------------------------------------------------------
# Summarisation (used by the deploy CLI under --enable-scheduling-v2)
# ---------------------------------------------------------------------------


def summarize_pending(diagnoses: Sequence[PodDiagnosis]) -> Dict[PendingReason, int]:
    """Count pending pods by reason. Non-pending pods are excluded."""
    counts: Dict[PendingReason, int] = {}
    for d in diagnoses:
        if d.reason is None:
            continue
        counts[d.reason] = counts.get(d.reason, 0) + 1
    return counts


def format_summary(diagnoses: Sequence[PodDiagnosis]) -> str:
    """One-line ``pending=N: REASON=k REASON=k`` digest for log output."""
    counts = summarize_pending(diagnoses)
    if not counts:
        return "pending=0"
    total = sum(counts.values())
    parts = [
        f"{reason.value}={count}"
        for reason, count in sorted(counts.items(), key=lambda kv: kv[0].value)
    ]
    return f"pending={total}: " + " ".join(parts)
