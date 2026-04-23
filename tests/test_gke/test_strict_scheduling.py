"""Tests for step 6 — strict-scheduling deploy + adopt-existing-job.

Two surfaces:

1. ``gke.lib.orchestration.deploy_dataset_strict`` — the candidate
   loop. We monkey-patch the kubectl/cluster helpers with fakes so
   the test exercises the decision tree without subprocess plumbing.

2. ``gke.orchestrator._adopt_existing_job`` and the helper
   ``_job_status`` — both pure functions over kubectl JSON shapes
   plus a tiny dependency on ``use_cluster``/``get_job_json`` that
   we monkey-patch.

Coverage targets the branches PLAN.md:330-343 specifies:

* WIN: full parallelism within deadline.
* CONFIG_ERROR / IMAGE_PULL_ERROR -> bail (raises StrictDeployError).
* CAPACITY_EXHAUSTED on every pending pod -> next candidate.
* TIMEOUT -> next candidate.
* All candidates exhausted, sleep_retry strategy -> retries until
  ``max_wait_hours``.
* All candidates exhausted, raise strategy -> returns spot_exhausted.
"""

from __future__ import annotations

import unittest
from typing import List, Optional, Tuple
from unittest import mock

from gke import orchestrator
from gke.lib import orchestration
from gke.lib.config import Config
from gke.lib.orchestration import (
    CandidateResult,
    StrictDeployError,
    deploy_dataset_strict,
)


def _make_cfg(
    *,
    clusters: List[Tuple[str, str]] = None,
    gpu_types: List[str] = None,
    total_pods: int = 4,
    attempt_deadline: int = 60,
    spot_exhausted_strategy: str = "raise",
    max_wait_hours: int = 1,
    spot_retry_interval: int = 1,
) -> Config:
    cfg = Config()
    cfg.project = "test-project"
    cfg.job_yaml = "/tmp/fake.yaml"
    cfg.clusters = clusters or [("c1", "us")]
    cfg.gpu_types = gpu_types or ["nvidia-h200-141gb"]
    cfg.total_pods = total_pods
    cfg.min_running = total_pods
    cfg.base_name = "regen-test"
    cfg.attempt_deadline = attempt_deadline
    cfg.spot_exhausted_strategy = spot_exhausted_strategy
    cfg.max_wait_hours = max_wait_hours
    cfg.spot_retry_interval = spot_retry_interval
    return cfg


class FakeAttemptScript:
    """Replays a scripted sequence of (CandidateResult, job_name) tuples
    for successive ``_attempt_candidate`` calls. Asserts exhaustion."""

    def __init__(self, calls: List[Tuple[CandidateResult, Optional[str]]]):
        self._calls = list(calls)
        self.invocations: List[Tuple[str, str, str]] = []

    def __call__(
        self, cfg: Config, dataset: str, cluster: str, region: str, gpu: str
    ) -> Tuple[CandidateResult, Optional[str]]:
        self.invocations.append((cluster, region, gpu))
        if not self._calls:
            raise AssertionError(
                f"FakeAttemptScript exhausted on call "
                f"{(cluster, region, gpu)} — provide more entries."
            )
        return self._calls.pop(0)


class StrictDeployHappyPath(unittest.TestCase):
    def test_first_candidate_wins(self) -> None:
        cfg = _make_cfg(
            clusters=[("c1", "us"), ("c2", "eu")],
            gpu_types=["nvidia-h200-141gb"],
        )
        script = FakeAttemptScript([(CandidateResult.WIN, "regen-test-ds-h200-us")])
        with (
            mock.patch.object(orchestration, "_attempt_candidate", script),
            mock.patch.object(orchestration, "_delete_quiet"),
        ):
            result = deploy_dataset_strict(cfg, "ds")
        self.assertTrue(result.succeeded)
        self.assertEqual(result.winner, ("regen-test-ds-h200-us", "c1", "us"))
        self.assertEqual(result.winner_gpu, "nvidia-h200-141gb")
        self.assertEqual(script.invocations, [("c1", "us", "nvidia-h200-141gb")])

    def test_falls_through_to_second_candidate(self) -> None:
        cfg = _make_cfg(
            clusters=[("c1", "us"), ("c2", "eu")],
            gpu_types=["nvidia-h200-141gb"],
        )
        script = FakeAttemptScript(
            [
                (CandidateResult.CAPACITY_EXHAUSTED, "regen-test-ds-h200-us"),
                (CandidateResult.WIN, "regen-test-ds-h200-eu"),
            ]
        )
        with (
            mock.patch.object(orchestration, "_attempt_candidate", script),
            mock.patch.object(orchestration, "_delete_quiet") as deleter,
        ):
            result = deploy_dataset_strict(cfg, "ds")
        self.assertTrue(result.succeeded)
        self.assertEqual(result.winner[1], "c2")
        # The losing job gets deleted.
        deleter.assert_called_with("regen-test-ds-h200-us")

    def test_outer_loop_iterates_gpu_first(self) -> None:
        """``--gpu-types h200,h100`` means try h200 across all clusters
        before falling back to h100. Documented at gke/lib/orchestration:
        _candidate_iter."""
        cfg = _make_cfg(
            clusters=[("c1", "us"), ("c2", "eu")],
            gpu_types=["nvidia-h200-141gb", "nvidia-h100-80gb"],
        )
        script = FakeAttemptScript(
            [
                (CandidateResult.CAPACITY_EXHAUSTED, "j1"),
                (CandidateResult.CAPACITY_EXHAUSTED, "j2"),
                (CandidateResult.WIN, "j3"),
            ]
        )
        with (
            mock.patch.object(orchestration, "_attempt_candidate", script),
            mock.patch.object(orchestration, "_delete_quiet"),
        ):
            deploy_dataset_strict(cfg, "ds")
        self.assertEqual(
            script.invocations,
            [
                ("c1", "us", "nvidia-h200-141gb"),
                ("c2", "eu", "nvidia-h200-141gb"),
                ("c1", "us", "nvidia-h100-80gb"),
            ],
        )


class StrictDeployBailPath(unittest.TestCase):
    def test_config_error_raises(self) -> None:
        cfg = _make_cfg()
        script = FakeAttemptScript(
            [(CandidateResult.CONFIG_ERROR, "regen-test-ds-h200-us")]
        )
        with (
            mock.patch.object(orchestration, "_attempt_candidate", script),
            mock.patch.object(orchestration, "_delete_quiet") as deleter,
        ):
            with self.assertRaises(StrictDeployError) as cm:
                deploy_dataset_strict(cfg, "ds")
        self.assertIn("config_error", str(cm.exception))
        deleter.assert_called_with("regen-test-ds-h200-us")

    def test_image_pull_error_raises_without_trying_next(self) -> None:
        cfg = _make_cfg(
            clusters=[("c1", "us"), ("c2", "eu")],
            gpu_types=["nvidia-h200-141gb"],
        )
        script = FakeAttemptScript(
            [(CandidateResult.IMAGE_PULL_ERROR, "regen-test-ds-h200-us")]
        )
        with (
            mock.patch.object(orchestration, "_attempt_candidate", script),
            mock.patch.object(orchestration, "_delete_quiet"),
        ):
            with self.assertRaises(StrictDeployError):
                deploy_dataset_strict(cfg, "ds")
        # Only the first candidate was tried — the second cluster never
        # got a chance because the image error is global.
        self.assertEqual(len(script.invocations), 1)


class StrictDeploySpotExhausted(unittest.TestCase):
    def test_raise_strategy_returns_spot_exhausted(self) -> None:
        cfg = _make_cfg(
            clusters=[("c1", "us")],
            gpu_types=["nvidia-h200-141gb"],
            spot_exhausted_strategy="raise",
        )
        script = FakeAttemptScript([(CandidateResult.CAPACITY_EXHAUSTED, "j1")])
        with (
            mock.patch.object(orchestration, "_attempt_candidate", script),
            mock.patch.object(orchestration, "_delete_quiet"),
        ):
            result = deploy_dataset_strict(cfg, "ds")
        self.assertFalse(result.succeeded)
        self.assertTrue(result.spot_exhausted)

    def test_sleep_retry_then_succeed(self) -> None:
        # Round 1: all exhausted. Round 2: WIN on first candidate.
        cfg = _make_cfg(
            clusters=[("c1", "us")],
            gpu_types=["nvidia-h200-141gb"],
            spot_exhausted_strategy="sleep_retry",
            spot_retry_interval=0,  # no actual sleep
            max_wait_hours=1,
        )
        script = FakeAttemptScript(
            [
                (CandidateResult.CAPACITY_EXHAUSTED, "j1"),
                (CandidateResult.WIN, "j2"),
            ]
        )
        with (
            mock.patch.object(orchestration, "_attempt_candidate", script),
            mock.patch.object(orchestration, "_delete_quiet"),
            mock.patch.object(orchestration.time, "sleep"),
        ):
            result = deploy_dataset_strict(cfg, "ds")
        self.assertTrue(result.succeeded)
        # Failed-candidates list is reset between rounds, so only the
        # final round's record (which had no failures before WIN) is
        # what callers see.
        self.assertEqual(result.failed_candidates, [])

    def test_sleep_retry_caps_at_max_wait_hours(self) -> None:
        cfg = _make_cfg(
            clusters=[("c1", "us")],
            gpu_types=["nvidia-h200-141gb"],
            spot_exhausted_strategy="sleep_retry",
            spot_retry_interval=10,
            max_wait_hours=1,
        )
        script = FakeAttemptScript([(CandidateResult.CAPACITY_EXHAUSTED, "j1")] * 50)

        # Force the wall-clock to advance past max_wait_hours after the
        # first round so the loop bails on the cap rather than retrying
        # forever.
        wall = [0.0]

        def fake_monotonic() -> float:
            return wall[0]

        def fake_sleep(_secs: float) -> None:
            wall[0] += 3600 * 2  # jump 2 hours per "sleep"

        with (
            mock.patch.object(orchestration, "_attempt_candidate", script),
            mock.patch.object(orchestration, "_delete_quiet"),
            mock.patch.object(orchestration.time, "monotonic", fake_monotonic),
            mock.patch.object(orchestration.time, "sleep", fake_sleep),
        ):
            result = deploy_dataset_strict(cfg, "ds")

        self.assertFalse(result.succeeded)
        self.assertTrue(result.spot_exhausted)


# ---------------------------------------------------------------------------
# Adopt-existing-job
# ---------------------------------------------------------------------------


class JobStatusTests(unittest.TestCase):
    def test_complete(self) -> None:
        doc = {
            "status": {
                "conditions": [{"type": "Complete", "status": "True"}],
                "active": 0,
            }
        }
        self.assertEqual(orchestrator._job_status(doc), "Complete")

    def test_failed(self) -> None:
        doc = {
            "status": {
                "conditions": [{"type": "Failed", "status": "True"}],
            }
        }
        self.assertEqual(orchestrator._job_status(doc), "Failed")

    def test_active(self) -> None:
        doc = {"status": {"active": 4}}
        self.assertEqual(orchestrator._job_status(doc), "Active")

    def test_unknown_when_no_signal(self) -> None:
        self.assertEqual(orchestrator._job_status({"status": {}}), "Unknown")
        self.assertEqual(orchestrator._job_status({}), "Unknown")

    def test_condition_with_status_false_is_ignored(self) -> None:
        doc = {
            "status": {
                "conditions": [{"type": "Complete", "status": "False"}],
                "active": 4,
            }
        }
        # Condition isn't True, so we fall through to active count.
        self.assertEqual(orchestrator._job_status(doc), "Active")


class AdoptExistingJobTests(unittest.TestCase):
    def setUp(self) -> None:
        # Minimal RunState with a recorded deploy entry for "ds".
        from gke import state as state_mod

        self.state_mod = state_mod
        self.run_state = state_mod.initial_state(
            run_id="test-run",
            config_yaml="/tmp/fake.yaml",
            config_hash="sha256:0",
        )
        self.run_state.phases.deploy["ds"] = {
            "primary_job": {
                "cluster": "c1",
                "region": "us",
                "gpu": "nvidia-h200-141gb",
                "job_name": "regen-test-ds-h200-us",
                "total_shards": 4,
                "started_at": "now",
            }
        }

    def _cfg(self) -> orchestrator.OrchestratorConfig:
        return orchestrator.OrchestratorConfig(
            config_yaml="/tmp/fake.yaml",
            datasets=["ds"],
            output_dir="gs://b/p",
            project="test-project",
        )

    def _deploy_cfg(self) -> Config:
        return _make_cfg()

    def test_returns_none_when_no_record(self) -> None:
        self.run_state.phases.deploy.clear()
        with (
            mock.patch.object(orchestrator, "use_cluster", return_value=True),
            mock.patch.object(orchestrator, "get_job_json", return_value=None),
        ):
            self.assertIsNone(
                orchestrator._adopt_existing_job(
                    self._cfg(), self._deploy_cfg(), "ds", self.run_state
                )
            )

    def test_returns_none_when_kubectl_context_fails(self) -> None:
        with mock.patch.object(orchestrator, "use_cluster", return_value=False):
            self.assertIsNone(
                orchestrator._adopt_existing_job(
                    self._cfg(), self._deploy_cfg(), "ds", self.run_state
                )
            )

    def test_returns_none_when_job_missing_in_cluster(self) -> None:
        with (
            mock.patch.object(orchestrator, "use_cluster", return_value=True),
            mock.patch.object(orchestrator, "get_job_json", return_value=None),
        ):
            self.assertIsNone(
                orchestrator._adopt_existing_job(
                    self._cfg(), self._deploy_cfg(), "ds", self.run_state
                )
            )

    def test_returns_none_on_failed_job(self) -> None:
        failed_doc = {"status": {"conditions": [{"type": "Failed", "status": "True"}]}}
        with (
            mock.patch.object(orchestrator, "use_cluster", return_value=True),
            mock.patch.object(orchestrator, "get_job_json", return_value=failed_doc),
        ):
            self.assertIsNone(
                orchestrator._adopt_existing_job(
                    self._cfg(), self._deploy_cfg(), "ds", self.run_state
                )
            )

    def test_adopts_active_job(self) -> None:
        active_doc = {"status": {"active": 4}}
        with (
            mock.patch.object(orchestrator, "use_cluster", return_value=True),
            mock.patch.object(orchestrator, "get_job_json", return_value=active_doc),
        ):
            adopted = orchestrator._adopt_existing_job(
                self._cfg(), self._deploy_cfg(), "ds", self.run_state
            )
        self.assertIsNotNone(adopted)
        job, cluster, region, gpu, status = adopted  # type: ignore
        self.assertEqual(job, "regen-test-ds-h200-us")
        self.assertEqual(status, "Active")

    def test_adopts_complete_job(self) -> None:
        complete_doc = {
            "status": {"conditions": [{"type": "Complete", "status": "True"}]}
        }
        with (
            mock.patch.object(orchestrator, "use_cluster", return_value=True),
            mock.patch.object(orchestrator, "get_job_json", return_value=complete_doc),
        ):
            adopted = orchestrator._adopt_existing_job(
                self._cfg(), self._deploy_cfg(), "ds", self.run_state
            )
        self.assertIsNotNone(adopted)
        self.assertEqual(adopted[4], "Complete")  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
