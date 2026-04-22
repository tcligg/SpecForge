"""Unit tests for gke/state.py.

These tests do not touch GCS; ``StateStore`` is exercised against a
fake gsutil runner that simulates success / generation-mismatch /
not-found responses. A separate manual smoke test against a real GCS
bucket is documented in gke/PLAN.md.
"""

from __future__ import annotations

import datetime as _dt
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

# The repo root contains the ``gke`` package.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gke import state  # noqa: E402

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers(unittest.TestCase):
    def test_compute_config_hash_is_deterministic(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("name: foo\n")
            path = f.name
        try:
            h1 = state.compute_config_hash(path)
            h2 = state.compute_config_hash(path)
            self.assertEqual(h1, h2)
            self.assertTrue(h1.startswith("sha256:"))
            self.assertEqual(len(h1), len("sha256:") + 64)
        finally:
            Path(path).unlink()

    def test_compute_config_hash_changes_when_file_changes(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("name: foo\n")
            path = f.name
        try:
            h1 = state.compute_config_hash(path)
            with open(path, "w") as f:
                f.write("name: bar\n")
            h2 = state.compute_config_hash(path)
            self.assertNotEqual(h1, h2)
        finally:
            Path(path).unlink()

    def test_compute_run_id_format(self):
        # Inject a fixed datetime so the format check is deterministic.
        fixed = _dt.datetime(2026, 4, 22, 16, 30, 45, tzinfo=_dt.timezone.utc)
        run_id = state.compute_run_id("regen-gemma3-27b", now=fixed)
        self.assertEqual(run_id, "regen-gemma3-27b-20260422-163045")

    def test_parse_state_uri_accepts_well_formed(self):
        self.assertEqual(
            state.parse_state_uri("gs://bucket/path/state.json"),
            "gs://bucket/path/state.json",
        )

    def test_parse_state_uri_rejects_bad_forms(self):
        for bad in [
            "",
            "bucket/path",
            "gs://",
            "gs://bucket",  # no path component
            "gs://bucket/",  # empty path component
            "https://bucket/p",
        ]:
            self.assertIsNone(state.parse_state_uri(bad), msg=bad)


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip(unittest.TestCase):
    def _make_state(self) -> state.RunState:
        return state.initial_state(
            run_id="regen-foo-20260422-160000",
            config_yaml="gke/regen-foo.yaml",
            config_hash="sha256:abc123",
        )

    def test_initial_state_populates_timestamps(self):
        s = self._make_state()
        self.assertTrue(s.created_at)
        self.assertTrue(s.updated_at)
        self.assertEqual(s.created_at, s.updated_at)
        self.assertEqual(s.schema_version, state.SCHEMA_VERSION)

    def test_to_json_then_from_json_round_trip(self):
        s = self._make_state()
        # Populate every phase so we cover the dict-or-dataclass paths.
        s.phases.prepare["perfectblend"] = {
            "status": "ok",
            "uri": "gs://b/p/perfectblend.jsonl",
            "size": 1234,
            "n_lines": 50,
        }
        s.phases.build = state.BuildPhase(
            image_uri="us-central1-docker.pkg.dev/p/r/i:tag",
            git_sha="abc123",
            dirty=False,
            completed_at="2026-04-22T16:00:00Z",
        )
        s.phases.deploy["perfectblend"] = {
            "primary_job": {
                "cluster": "pyc-batch-eu",
                "region": "europe-west1",
                "gpu": "nvidia-h200-141gb",
                "job_name": "regen-foo-perfectblend-h200-europe-west1",
                "total_shards": 8,
                "started_at": "2026-04-22T16:01:00Z",
            }
        }
        s.phases.monitor["perfectblend"] = {
            "primary_job_status": "Active",
            "shard_status": {"0": "Complete", "1": "Pending"},
        }
        s.phases.merge["perfectblend"] = {
            "uri": "gs://b/p/perfectblend_regen.jsonl",
            "chunks_merged": 320,
            "samples": 160000,
        }

        payload = s.to_json()
        s2 = state.RunState.from_json(payload)

        self.assertEqual(s2.run_id, s.run_id)
        self.assertEqual(s2.config_hash, s.config_hash)
        self.assertEqual(s2.schema_version, s.schema_version)
        self.assertEqual(s2.phases.prepare, s.phases.prepare)
        self.assertEqual(s2.phases.build.image_uri, s.phases.build.image_uri)
        self.assertEqual(s2.phases.deploy, s.phases.deploy)
        self.assertEqual(s2.phases.monitor, s.phases.monitor)
        self.assertEqual(s2.phases.merge, s.phases.merge)

    def test_from_dict_handles_missing_phase_keys(self):
        # A minimal payload without 'phases' should still produce a
        # usable RunState with empty containers.
        minimal = {
            "run_id": "x",
            "config_yaml": "y.yaml",
            "config_hash": "sha256:0",
            "schema_version": 1,
        }
        s = state.RunState.from_dict(minimal)
        self.assertEqual(s.phases.prepare, {})
        self.assertEqual(s.phases.build.image_uri, "")
        self.assertEqual(s.phases.deploy, {})

    def test_build_phase_from_dict_with_partial_input(self):
        bp = state.BuildPhase.from_dict({"image_uri": "img"})
        self.assertEqual(bp.image_uri, "img")
        self.assertEqual(bp.git_sha, "")
        self.assertFalse(bp.dirty)


# ---------------------------------------------------------------------------
# StateStore against a fake gsutil
# ---------------------------------------------------------------------------


class FakeGsutil:
    """Records every gsutil invocation; replays canned responses.

    The StateStore drives gsutil with these subcommands:
      stat <uri>            -> Generation: <int> in stdout
      cat <uri>             -> body in stdout
      cp <local> <uri>      -> writes to in-memory store
      cp -h <hdr> ...       -> as above, but checks generation precondition
    """

    def __init__(self):
        # uri -> (body, generation)
        self.objects: dict[str, tuple[str, int]] = {}
        self._next_generation = 1000
        self.calls: List[List[str]] = []

    def __call__(self, *args, stdin=None) -> state._GsutilResult:
        self.calls.append(list(args))
        # Strip header flags into a dict
        headers: dict[str, str] = {}
        positional: List[str] = []
        i = 0
        while i < len(args):
            if args[i] == "-h" and i + 1 < len(args):
                k, _, v = args[i + 1].partition(":")
                headers[k.strip().lower()] = v.strip()
                i += 2
            else:
                positional.append(args[i])
                i += 1
        if not positional:
            return state._GsutilResult(2, "", "no command\n")
        cmd, *rest = positional
        if cmd == "stat":
            uri = rest[0]
            if uri not in self.objects:
                return state._GsutilResult(1, "", "No URLs matched\n")
            _body, gen = self.objects[uri]
            return state._GsutilResult(0, f"Generation:    {gen}\n", "")
        if cmd == "cat":
            uri = rest[0]
            if uri not in self.objects:
                return state._GsutilResult(1, "", "No URLs matched\n")
            body, _gen = self.objects[uri]
            return state._GsutilResult(0, body, "")
        if cmd == "cp":
            src, dst = rest
            with open(src, "r") as f:
                body = f.read()
            required_gen = headers.get("x-goog-if-generation-match")
            if required_gen is not None:
                current_gen = self.objects.get(dst, (None, -1))[1]
                if int(required_gen) != current_gen:
                    return state._GsutilResult(
                        1,
                        "",
                        "PreconditionException: 412 At least one of "
                        "the pre-conditions you specified did not hold.\n",
                    )
            self._next_generation += 1
            self.objects[dst] = (body, self._next_generation)
            return state._GsutilResult(0, "", "")
        return state._GsutilResult(2, "", f"unknown command {cmd}\n")


class TestStateStoreRead(unittest.TestCase):
    def test_read_returns_none_when_object_missing(self):
        runner = FakeGsutil()
        store = state.StateStore("gs://bucket/state.json", runner=runner)
        self.assertIsNone(store.read())

    def test_read_required_raises_when_missing(self):
        runner = FakeGsutil()
        store = state.StateStore("gs://bucket/state.json", runner=runner)
        with self.assertRaises(state.StateNotFound):
            store.read(required=True)

    def test_read_returns_state_after_write(self):
        runner = FakeGsutil()
        store = state.StateStore("gs://bucket/state.json", runner=runner)
        s = state.initial_state("run-1", "y.yaml", "sha256:x")
        store.write(s)
        loaded = store.read()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "run-1")


class TestStateStoreWrite(unittest.TestCase):
    def test_write_unconditional_succeeds(self):
        runner = FakeGsutil()
        store = state.StateStore("gs://bucket/state.json", runner=runner)
        s = state.initial_state("run-1", "y.yaml", "sha256:x")
        store.write(s)
        # The underlying object now contains our payload.
        body, _gen = runner.objects["gs://bucket/state.json"]
        loaded = state.RunState.from_json(body)
        self.assertEqual(loaded.run_id, "run-1")

    def test_write_overwrites_existing(self):
        runner = FakeGsutil()
        store = state.StateStore("gs://bucket/state.json", runner=runner)
        store.write(state.initial_state("run-1", "y.yaml", "sha256:x"))
        store.write(state.initial_state("run-2", "y.yaml", "sha256:x"))
        loaded = store.read()
        self.assertEqual(loaded.run_id, "run-2")


class TestStateStoreUpdate(unittest.TestCase):
    def _store_with_initial(self) -> tuple[state.StateStore, FakeGsutil]:
        runner = FakeGsutil()
        store = state.StateStore("gs://bucket/state.json", runner=runner)
        store.write(state.initial_state("run-1", "y.yaml", "sha256:x"))
        return store, runner

    def test_update_modifies_state_and_uses_precondition(self):
        store, runner = self._store_with_initial()

        def mutate(s: state.RunState) -> state.RunState:
            s.phases.build.image_uri = "img:tag"
            return s

        result = store.update(mutate)
        self.assertEqual(result.phases.build.image_uri, "img:tag")
        loaded = store.read()
        self.assertEqual(loaded.phases.build.image_uri, "img:tag")

        # The cp call after the read must have included the
        # precondition header.
        cp_calls = [c for c in runner.calls if "cp" in c]
        # There were two cp calls: the initial write() and the update().
        # Initial write has no -h; update must.
        self.assertEqual(len(cp_calls), 2)
        self.assertNotIn("-h", cp_calls[0])
        self.assertIn("-h", cp_calls[1])

    def test_update_raises_when_no_existing_state(self):
        runner = FakeGsutil()
        store = state.StateStore("gs://bucket/state.json", runner=runner)
        with self.assertRaises(state.StateNotFound):
            store.update(lambda s: s)

    def test_update_retries_on_generation_conflict(self):
        store, runner = self._store_with_initial()
        original_runner = runner

        # Wrap the runner to inject one synthetic 412 on the first cp
        # after the read, then let real behavior take over.
        injected = {"injected": False}

        def runner_with_one_conflict(*args, stdin=None):
            res = original_runner(*args, stdin=stdin)
            if (
                not injected["injected"]
                and "cp" in args
                and any(a.startswith("-h") for a in args) is False
                and any("if-generation-match" in (a or "") for a in args)
            ):
                injected["injected"] = True
                return state._GsutilResult(
                    1,
                    "",
                    "PreconditionException: 412 At least one of the "
                    "pre-conditions you specified did not hold.\n",
                )
            return res

        store_with_injection = state.StateStore(
            "gs://bucket/state.json",
            runner=runner_with_one_conflict,
            retry_backoff_s=0,  # speed up test
        )
        # Seed the FakeGsutil-backed object via the injected runner so
        # cat/stat hit the same in-memory store.
        # (The previous self._store_with_initial already populated it.)

        def mutate(s: state.RunState) -> state.RunState:
            s.phases.build.image_uri = "after-conflict"
            return s

        # The mutation should still succeed because update() retries.
        result = store_with_injection.update(mutate)
        self.assertEqual(result.phases.build.image_uri, "after-conflict")

    def test_update_gives_up_after_max_retries(self):
        runner = FakeGsutil()
        # Seed an existing object so update() can read it.
        runner.objects["gs://bucket/state.json"] = (
            state.initial_state("run-1", "y.yaml", "sha256:x").to_json(),
            42,
        )

        # Wrap the runner so every cp returns a precondition failure.
        def always_conflict(*args, stdin=None):
            if "cp" in args:
                return state._GsutilResult(
                    1,
                    "",
                    "PreconditionException: 412 At least one of the "
                    "pre-conditions you specified did not hold.\n",
                )
            # Defer non-cp ops to the real fake (stat / cat).
            return runner(*args, stdin=stdin)

        store = state.StateStore(
            "gs://bucket/state.json",
            runner=always_conflict,
            max_retries=3,
            retry_backoff_s=0,
        )
        with self.assertRaises(state.StateConflict):
            store.update(lambda s: s)


class TestStateStoreInit(unittest.TestCase):
    def test_rejects_bad_uri_at_construction(self):
        with self.assertRaises(state.StateError):
            state.StateStore("not-a-gcs-uri")


if __name__ == "__main__":
    unittest.main()
