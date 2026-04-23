"""Tests for step 7 — Phase D.5 rescue.

Two surfaces:

1. ``rescue.compute_missing_chunks`` — pure-ish (subprocess-mocked)
   computation of which chunk IDs lack a ``.done`` marker.
2. ``rescue._write_rescue_yaml`` — string transform on the YAML;
   tested by round-tripping a fixture.

The ``phase_d5_rescue_dataset`` orchestration is covered indirectly:
its decision points (missing math, attempt cap, recording state) are
small enough to assert via the underlying helpers + a smoke test
that exercises the loop with mocked submission/monitor calls.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import List
from unittest import mock

from gke import rescue
from gke import state as state_mod
from gke.lib.config import Config


def _make_state_with_prepare(n_lines: int) -> state_mod.RunState:
    s = state_mod.initial_state(
        run_id="test-run",
        config_yaml="/tmp/fake.yaml",
        config_hash="sha256:0",
    )
    s.phases.prepare["ds"] = {
        "status": "ok",
        "uri": "gs://b/p/ds_train.jsonl",
        "size": 12345,
        "n_lines": n_lines,
        "completed_at": "now",
    }
    return s


def _make_cfg(*, output_dir: str = "gs://b/p", chunk_size: int = 100) -> Config:
    cfg = Config()
    cfg.job_yaml = "/tmp/fake.yaml"
    cfg.output_dir = output_dir
    cfg.base_name = "regen-test"
    cfg.chunk_size = chunk_size
    cfg.max_rescue_attempts = 3
    cfg.max_rescue_shards = 4
    cfg.monitor_deadline = 60
    return cfg


# ---------------------------------------------------------------------------


class ComputeMissingChunksTests(unittest.TestCase):
    def test_all_done_returns_empty(self) -> None:
        state = _make_state_with_prepare(n_lines=300)  # 3 chunks @ size 100
        cfg = _make_cfg()
        with mock.patch.object(rescue, "_list_done_markers", return_value=[0, 1, 2]):
            self.assertEqual(rescue.compute_missing_chunks(state, cfg, "ds"), [])

    def test_some_missing(self) -> None:
        state = _make_state_with_prepare(n_lines=500)  # 5 chunks
        cfg = _make_cfg()
        with mock.patch.object(rescue, "_list_done_markers", return_value=[0, 2, 4]):
            self.assertEqual(rescue.compute_missing_chunks(state, cfg, "ds"), [1, 3])

    def test_partial_chunk_at_tail_counted(self) -> None:
        # 250 lines / chunk_size 100 = 3 chunks (last one is partial).
        state = _make_state_with_prepare(n_lines=250)
        cfg = _make_cfg()
        with mock.patch.object(rescue, "_list_done_markers", return_value=[0, 1]):
            self.assertEqual(rescue.compute_missing_chunks(state, cfg, "ds"), [2])

    def test_no_done_markers_means_all_missing(self) -> None:
        state = _make_state_with_prepare(n_lines=300)
        cfg = _make_cfg()
        with mock.patch.object(rescue, "_list_done_markers", return_value=[]):
            self.assertEqual(rescue.compute_missing_chunks(state, cfg, "ds"), [0, 1, 2])

    def test_chunks_dir_unresolvable_means_all_missing(self) -> None:
        # _resolve_output_dir returns None when output_dir is unset and
        # the YAML can't be parsed. Simulate by mocking
        # gke.lib.merge.chunks_dir_uri to return None.
        from gke.lib import merge as merge_mod

        state = _make_state_with_prepare(n_lines=300)
        cfg = _make_cfg(output_dir="")
        with mock.patch.object(merge_mod, "chunks_dir_uri", return_value=None):
            self.assertEqual(rescue.compute_missing_chunks(state, cfg, "ds"), [0, 1, 2])

    def test_missing_n_lines_raises(self) -> None:
        state = state_mod.initial_state(
            run_id="r", config_yaml="x", config_hash="sha:0"
        )
        cfg = _make_cfg()
        with self.assertRaises(ValueError) as cm:
            rescue.compute_missing_chunks(state, cfg, "ds")
        self.assertIn("n_lines", str(cm.exception))

    def test_zero_chunk_size_raises(self) -> None:
        state = _make_state_with_prepare(n_lines=300)
        cfg = _make_cfg(chunk_size=0)
        with self.assertRaises(ValueError) as cm:
            rescue.compute_missing_chunks(state, cfg, "ds")
        self.assertIn("chunk_size", str(cm.exception))


# ---------------------------------------------------------------------------


class WriteRescueYamlTests(unittest.TestCase):
    """``_write_rescue_yaml`` is a string transform; round-trip a real
    fixture to make sure we touch the right keys without disturbing
    anything else."""

    def setUp(self) -> None:
        # Use the real gemma3-27b YAML as the fixture so we exercise
        # the same shape the orchestrator would see in production.
        self.src = "gke/regen-gemma3-27b.yaml"
        with open(self.src) as f:
            self.original = f.read()
        self.tmpdir = tempfile.mkdtemp(prefix="rescue-test-")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_to(self, *, shards: int, ids: List[int]) -> str:
        dst = os.path.join(self.tmpdir, "rescue.yaml")
        rescue._write_rescue_yaml(
            src_yaml=self.src,
            dst_yaml=dst,
            base_name="regen-gemma3-27b",
            rescue_shards=shards,
            chunk_ids=ids,
        )
        with open(dst) as f:
            return f.read()

    def test_metadata_name_suffixed(self) -> None:
        out = self._patch_to(shards=2, ids=[5])
        self.assertIn("name: regen-gemma3-27b-rescue", out)
        self.assertNotIn("name: regen-gemma3-27b\n", out)

    def test_completions_and_parallelism_overridden(self) -> None:
        out = self._patch_to(shards=2, ids=[5])
        self.assertIn("completions: 2", out)
        self.assertIn("parallelism: 2", out)
        self.assertNotIn("completions: 8", out)
        self.assertNotIn("parallelism: 8", out)

    def test_extra_regen_args_injected(self) -> None:
        out = self._patch_to(shards=3, ids=[12, 17, 89])
        self.assertIn("EXTRA_REGEN_ARGS", out)
        self.assertIn("--chunk-ids 12,17,89", out)

    def test_extra_regen_args_replaces_existing(self) -> None:
        # Build a fixture with EXTRA_REGEN_ARGS already present.
        src_with_extra = os.path.join(self.tmpdir, "src.yaml")
        with open(src_with_extra, "w") as f:
            f.write(
                self.original.replace(
                    "        - name: IS_REASONING_MODEL",
                    "        - name: EXTRA_REGEN_ARGS\n"
                    '          value: "--something old"\n'
                    "        - name: IS_REASONING_MODEL",
                    1,
                )
            )
        dst = os.path.join(self.tmpdir, "out.yaml")
        rescue._write_rescue_yaml(
            src_yaml=src_with_extra,
            dst_yaml=dst,
            base_name="regen-gemma3-27b",
            rescue_shards=2,
            chunk_ids=[42],
        )
        with open(dst) as f:
            out = f.read()
        self.assertIn("--chunk-ids 42", out)
        self.assertNotIn("--something old", out)


# ---------------------------------------------------------------------------


class PhaseD5LoopTests(unittest.TestCase):
    """Smoke-test the rescue loop's decision points."""

    def test_returns_immediately_when_nothing_missing(self) -> None:
        state = _make_state_with_prepare(n_lines=300)
        cfg = _make_cfg()
        store = mock.MagicMock()
        # Stub all the I/O: missing computes to [], so we never submit.
        with mock.patch.object(rescue, "compute_missing_chunks", return_value=[]):
            rescue.phase_d5_rescue_dataset(
                cfg,
                state,
                store,
                "ds",
                src_yaml="/tmp/fake.yaml",
                base_name="regen-test",
            )
        # State recorded all_chunks_done.
        store.update.assert_called()  # at least one update for the 'done' write

    def test_loops_until_attempts_exhausted_then_raises(self) -> None:
        state = _make_state_with_prepare(n_lines=300)
        cfg = _make_cfg()
        cfg.max_rescue_attempts = 2
        store = mock.MagicMock()
        store.read.return_value = state

        # Always missing two chunks; rescue submission "fails" each time.
        with (
            mock.patch.object(rescue, "compute_missing_chunks", return_value=[1, 2]),
            mock.patch.object(rescue, "_write_rescue_yaml"),
            mock.patch.object(
                rescue,
                "deploy_dataset_strict",
                return_value=mock.MagicMock(succeeded=False, spot_exhausted=True),
            ),
        ):
            with self.assertRaises(rescue.RescueError) as cm:
                rescue.phase_d5_rescue_dataset(
                    cfg,
                    state,
                    store,
                    "ds",
                    src_yaml="/tmp/fake.yaml",
                    base_name="regen-test",
                )
        self.assertIn("2 chunks still missing", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
