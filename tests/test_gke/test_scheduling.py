"""Unit tests for gke/scheduling.py.

Step 5 of gke/PLAN.md ships a pure classifier whose inputs are the
shapes returned by ``kubectl get pod -o json`` and ``kubectl get
events -o json``. These tests build minimal pod/event dicts that
exercise each branch of the decision matrix at PLAN.md:307-313.

We intentionally don't import kubectl or hit the network; the
``diagnose_job`` orchestration helper has its own thin integration
that's covered in the smoke section by monkey-patching the
``gke.lib.k8s`` getters.
"""

from __future__ import annotations

import datetime as dt
import unittest
from typing import Any, Dict, List

from gke import scheduling
from gke.scheduling import (
    SCALE_UP_FRESH_SECONDS,
    PendingReason,
    PodDiagnosis,
    classify_pod,
    diagnose_job,
    format_summary,
    summarize_pending,
)


def _ts(now: dt.datetime, *, seconds_ago: int) -> str:
    """Render a kubectl-style ISO-8601 (...Z) timestamp."""
    return (now - dt.timedelta(seconds=seconds_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pending_pod(
    name: str,
    *,
    sched_message: str = "",
    waiting_reasons: List[str] = None,
) -> Dict[str, Any]:
    """Build a minimal Pending pod with the requested signals."""
    container_statuses = []
    for reason in waiting_reasons or []:
        container_statuses.append(
            {"state": {"waiting": {"reason": reason, "message": "x"}}}
        )
    conditions = []
    if sched_message:
        conditions.append(
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "Unschedulable",
                "message": sched_message,
            }
        )
    return {
        "metadata": {"name": name},
        "status": {
            "phase": "Pending",
            "conditions": conditions,
            "containerStatuses": container_statuses,
        },
    }


def _running_pod(name: str) -> Dict[str, Any]:
    return {"metadata": {"name": name}, "status": {"phase": "Running"}}


def _scale_up_event(now: dt.datetime, *, age_s: int) -> Dict[str, Any]:
    return {
        "reason": "TriggeredScaleUp",
        "message": "pod triggered scale-up",
        "lastTimestamp": _ts(now, seconds_ago=age_s),
    }


# ---------------------------------------------------------------------------


class ClassifyPodTests(unittest.TestCase):
    def setUp(self) -> None:
        # Anchor "now" so all event ages are deterministic.
        self.now = dt.datetime(2026, 5, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

    def test_running_pod_has_no_reason(self) -> None:
        diag = classify_pod(_running_pod("p"), [], now=self.now)
        self.assertEqual(diag.phase, "Running")
        self.assertIsNone(diag.reason)

    def test_capacity_wait_when_scaleup_fresh(self) -> None:
        pod = _pending_pod(
            "p",
            sched_message="0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
        )
        events = [_scale_up_event(self.now, age_s=60)]
        diag = classify_pod(pod, events, now=self.now)
        self.assertEqual(diag.reason, PendingReason.CAPACITY_WAIT)
        self.assertEqual(diag.last_event_age_seconds, 60)

    def test_capacity_exhausted_when_scaleup_stale(self) -> None:
        pod = _pending_pod(
            "p",
            sched_message="0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
        )
        # Older than the freshness window.
        stale_age = SCALE_UP_FRESH_SECONDS + 60
        events = [_scale_up_event(self.now, age_s=stale_age)]
        diag = classify_pod(pod, events, now=self.now)
        self.assertEqual(diag.reason, PendingReason.CAPACITY_EXHAUSTED)

    def test_capacity_exhausted_when_no_scaleup_event(self) -> None:
        pod = _pending_pod(
            "p",
            sched_message="0/8 nodes are available: 8 Insufficient nvidia.com/gpu.",
        )
        diag = classify_pod(pod, [], now=self.now)
        self.assertEqual(diag.reason, PendingReason.CAPACITY_EXHAUSTED)

    def test_image_pull_takes_precedence_over_scheduling(self) -> None:
        pod = _pending_pod(
            "p",
            sched_message="0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
            waiting_reasons=["ImagePullBackOff"],
        )
        diag = classify_pod(pod, [], now=self.now)
        self.assertEqual(diag.reason, PendingReason.IMAGE_PULL_ERROR)

    def test_err_image_pull_classified(self) -> None:
        pod = _pending_pod("p", waiting_reasons=["ErrImagePull"])
        diag = classify_pod(pod, [], now=self.now)
        self.assertEqual(diag.reason, PendingReason.IMAGE_PULL_ERROR)

    def test_invalid_image_name_classified(self) -> None:
        pod = _pending_pod("p", waiting_reasons=["InvalidImageName"])
        diag = classify_pod(pod, [], now=self.now)
        self.assertEqual(diag.reason, PendingReason.IMAGE_PULL_ERROR)

    def test_config_error_volume_message(self) -> None:
        pod = _pending_pod(
            "p",
            sched_message=(
                '0/3 nodes are available: 3 persistentvolumeclaim "data" not found.'
            ),
        )
        diag = classify_pod(pod, [], now=self.now)
        self.assertEqual(diag.reason, PendingReason.CONFIG_ERROR)

    def test_config_error_secret_missing(self) -> None:
        pod = _pending_pod("p", sched_message='secret "hf-token" not found')
        diag = classify_pod(pod, [], now=self.now)
        self.assertEqual(diag.reason, PendingReason.CONFIG_ERROR)

    def test_config_error_node_affinity(self) -> None:
        pod = _pending_pod(
            "p",
            sched_message="0/3 nodes are available: 3 node(s) didn't match Pod's node affinity.",
        )
        diag = classify_pod(pod, [], now=self.now)
        self.assertEqual(diag.reason, PendingReason.CONFIG_ERROR)

    def test_unknown_when_no_signal_matches(self) -> None:
        pod = _pending_pod("p", sched_message="something completely new")
        diag = classify_pod(pod, [], now=self.now)
        self.assertEqual(diag.reason, PendingReason.UNKNOWN)

    def test_scaleup_event_without_timestamp_treated_as_fresh(self) -> None:
        # Defensive: if kubectl drops the timestamps for some reason, we
        # prefer waiting one extra cycle to fast-failing on bad data.
        pod = _pending_pod(
            "p",
            sched_message="0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
        )
        events = [{"reason": "TriggeredScaleUp", "message": "no ts"}]
        diag = classify_pod(pod, events, now=self.now)
        self.assertEqual(diag.reason, PendingReason.CAPACITY_WAIT)

    def test_pod_with_no_metadata_is_handled(self) -> None:
        # Defensive: kubectl returns well-formed JSON but a future API
        # change could move name elsewhere. Make sure we don't KeyError.
        diag = classify_pod({"status": {"phase": "Pending"}}, [], now=self.now)
        self.assertEqual(diag.pod_name, "<unknown>")
        self.assertEqual(diag.reason, PendingReason.UNKNOWN)

    def test_eventtime_preferred_over_lasttimestamp(self) -> None:
        pod = _pending_pod(
            "p",
            sched_message="0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
        )
        events = [
            {
                "reason": "TriggeredScaleUp",
                # eventTime says fresh, lastTimestamp says stale.
                "eventTime": _ts(self.now, seconds_ago=10),
                "lastTimestamp": _ts(
                    self.now, seconds_ago=SCALE_UP_FRESH_SECONDS + 100
                ),
            }
        ]
        diag = classify_pod(pod, events, now=self.now)
        self.assertEqual(diag.reason, PendingReason.CAPACITY_WAIT)


class SummariseTests(unittest.TestCase):
    def test_summary_counts_pending_only(self) -> None:
        diags = [
            PodDiagnosis("p1", "Pending", PendingReason.CAPACITY_WAIT, "", None),
            PodDiagnosis("p2", "Pending", PendingReason.CAPACITY_WAIT, "", None),
            PodDiagnosis("p3", "Pending", PendingReason.IMAGE_PULL_ERROR, "", None),
            PodDiagnosis("p4", "Running", None, "", None),
        ]
        counts = summarize_pending(diags)
        self.assertEqual(counts[PendingReason.CAPACITY_WAIT], 2)
        self.assertEqual(counts[PendingReason.IMAGE_PULL_ERROR], 1)
        self.assertNotIn(None, counts)
        self.assertEqual(sum(counts.values()), 3)

    def test_format_summary_empty(self) -> None:
        self.assertEqual(format_summary([]), "pending=0")

    def test_format_summary_sorted(self) -> None:
        diags = [
            PodDiagnosis("p1", "Pending", PendingReason.IMAGE_PULL_ERROR, "", None),
            PodDiagnosis("p2", "Pending", PendingReason.CAPACITY_WAIT, "", None),
        ]
        out = format_summary(diags)
        # Sorted by enum value; capacity_wait < image_pull_error.
        self.assertEqual(out, "pending=2: capacity_wait=1 image_pull_error=1")


class DiagnoseJobTests(unittest.TestCase):
    """Thin integration test: monkey-patch the kubectl getters."""

    def test_diagnose_job_aggregates_per_pod(self) -> None:
        from gke.lib import k8s

        pods = [
            _pending_pod(
                "shard-0",
                sched_message="0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
            ),
            _running_pod("shard-1"),
        ]
        events_by_pod = {
            "shard-0": [],  # no scale-up -> EXHAUSTED
            "shard-1": [],
        }

        orig_pods = k8s.get_pods_json
        orig_events = k8s.get_pod_events
        try:
            k8s.get_pods_json = lambda job: pods  # type: ignore
            k8s.get_pod_events = lambda pod: events_by_pod.get(pod, [])  # type: ignore
            result = diagnose_job("regen-foo")
        finally:
            k8s.get_pods_json = orig_pods
            k8s.get_pod_events = orig_events

        self.assertEqual(len(result), 2)
        by_name = {d.pod_name: d for d in result}
        self.assertEqual(by_name["shard-0"].reason, PendingReason.CAPACITY_EXHAUSTED)
        self.assertIsNone(by_name["shard-1"].reason)

    def test_diagnose_job_empty_when_no_pods(self) -> None:
        from gke.lib import k8s

        orig_pods = k8s.get_pods_json
        try:
            k8s.get_pods_json = lambda job: []  # type: ignore
            self.assertEqual(diagnose_job("regen-nopods"), [])
        finally:
            k8s.get_pods_json = orig_pods


if __name__ == "__main__":
    unittest.main()
