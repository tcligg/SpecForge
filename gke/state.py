"""GCS-backed run state for the GKE regen pipeline.

Step 3 of the rebuild plan in gke/PLAN.md. This module exists so that
step 4+ can resume work across runs. In step 3 itself, the module is
wired into ``gke/deploy.py`` as a *pure observer*: when ``--state-uri``
is set, ``cmd_execute`` writes state at each phase boundary but never
reads it. Read paths are exercised by step 4 onward.

Schema (v1, mirrored from PLAN.md "State schema" section):

    {
      "run_id": "<base_name>-<UTC timestamp>",
      "config_yaml": "gke/regen-<model>.yaml",
      "config_hash": "sha256:...",
      "schema_version": 1,
      "created_at": "<ISO 8601>",
      "updated_at": "<ISO 8601>",
      "phases": {
        "prepare": { "<dataset>": {...} },
        "build":   {...},
        "deploy":  { "<dataset>": {...} },
        "monitor": { "<dataset>": {...} },
        "rescue":  { "<dataset>": {...} },
        "merge":   { "<dataset>": {...} }
      }
    }

The top level (``RunState`` and the six phase containers) is dataclassed
for type safety; per-dataset entries are plain dicts because the keys
(dataset names) are open-ended and the per-phase value shapes are still
evolving.

No new Python deps. All GCS I/O goes through the ``gsutil`` CLI with
``If-Generation-Match`` preconditions for atomic update().
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp, second precision (good enough for state)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_config_hash(config_path: str) -> str:
    """sha256 of the YAML file contents, prefixed with 'sha256:'.

    Lets the orchestrator detect 'you edited the YAML between runs' and
    refuse to silently resume a run against a different config.
    """
    h = hashlib.sha256()
    with open(config_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def compute_run_id(prefix: str, *, now: Optional[_dt.datetime] = None) -> str:
    """``<prefix>-YYYYMMDD-HHMMSS`` (UTC).

    The prefix typically comes from ``Config.base_name`` (which is the
    YAML's ``metadata.name``). ``now`` is injectable for deterministic
    tests; production callers should leave it None.
    """
    ts = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}"


def parse_state_uri(uri: str) -> Optional[str]:
    """Validate that uri looks like ``gs://bucket/path``. Returns the URI
    on success, None otherwise. We deliberately don't raise here so the
    --state-uri flag can be 'soft' in step 3 (observer-only mode).
    """
    if not uri:
        return None
    if not uri.startswith("gs://"):
        return None
    rest = uri[len("gs://") :]
    if "/" not in rest or rest.endswith("/"):
        return None
    return uri


# ---------------------------------------------------------------------------
# Dataclass schema
# ---------------------------------------------------------------------------


@dataclass
class BuildPhase:
    """Outcome of phase B (image build/push)."""

    image_uri: str = ""
    git_sha: str = ""
    dirty: bool = False
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "BuildPhase":
        if not d:
            return cls()
        return cls(
            image_uri=d.get("image_uri", ""),
            git_sha=d.get("git_sha", ""),
            dirty=bool(d.get("dirty", False)),
            completed_at=d.get("completed_at", ""),
        )


@dataclass
class Phases:
    """Container for the six per-phase records.

    Per-dataset entries (everything except ``build``) stay as plain dicts;
    the orchestrator/rescue modules will define richer types as the
    schema settles. Dict values for missing entries default to ``{}``.
    """

    prepare: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    build: BuildPhase = field(default_factory=BuildPhase)
    deploy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    monitor: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rescue: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    merge: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prepare": dict(self.prepare),
            "build": self.build.to_dict(),
            "deploy": dict(self.deploy),
            "monitor": dict(self.monitor),
            "rescue": dict(self.rescue),
            "merge": dict(self.merge),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Phases":
        d = d or {}
        return cls(
            prepare=dict(d.get("prepare") or {}),
            build=BuildPhase.from_dict(d.get("build")),
            deploy=dict(d.get("deploy") or {}),
            monitor=dict(d.get("monitor") or {}),
            rescue=dict(d.get("rescue") or {}),
            merge=dict(d.get("merge") or {}),
        )


@dataclass
class RunState:
    """Top-level state document for one run."""

    run_id: str
    config_yaml: str
    config_hash: str
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    updated_at: str = ""
    phases: Phases = field(default_factory=Phases)

    def to_json(self, *, indent: int = 2) -> str:
        d = {
            "run_id": self.run_id,
            "config_yaml": self.config_yaml,
            "config_hash": self.config_hash,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phases": self.phases.to_dict(),
        }
        return json.dumps(d, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "RunState":
        d = json.loads(payload)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunState":
        sv = int(d.get("schema_version", SCHEMA_VERSION))
        if sv != SCHEMA_VERSION:
            # Forward compatibility is the orchestrator's job, not ours.
            # We accept and pass through; callers can branch on
            # ``state.schema_version`` if they need migration.
            pass
        return cls(
            run_id=d["run_id"],
            config_yaml=d["config_yaml"],
            config_hash=d["config_hash"],
            schema_version=sv,
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            phases=Phases.from_dict(d.get("phases")),
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StateError(RuntimeError):
    """Base class for state I/O errors."""


class StateConflict(StateError):
    """Raised when a write fails the If-Generation-Match precondition
    after the configured number of retries."""


class StateNotFound(StateError):
    """Raised when a read targets a state object that does not exist
    (and the caller asked for required=True)."""


# ---------------------------------------------------------------------------
# Subprocess plumbing (gsutil)
# ---------------------------------------------------------------------------


@dataclass
class _GsutilResult:
    returncode: int
    stdout: str
    stderr: str


def _run_gsutil(*args: str, stdin: Optional[bytes] = None) -> _GsutilResult:
    """Run gsutil with the given args; return captured output.

    Kept private because callers should go through the higher-level
    StateStore API. Exposed via parameter ``_runner`` on StateStore for
    test injection.
    """
    proc = subprocess.run(
        ["gsutil"] + list(args),
        input=stdin,
        capture_output=True,
    )
    return _GsutilResult(
        returncode=proc.returncode,
        stdout=proc.stdout.decode("utf-8", errors="replace"),
        stderr=proc.stderr.decode("utf-8", errors="replace"),
    )


# Generation lines look like:   "Generation:        1693846200123456"
_GEN_RE = re.compile(r"^\s*Generation:\s*(\d+)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------------


class StateStore:
    """Read/write a single ``state.json`` object in GCS.

    All writes are atomic via gsutil's ``-h x-goog-if-generation-match:N``
    precondition. ``update()`` performs read-modify-write with bounded
    retry on conflict.

    Thread/process safety: concurrent writers from any host coordinate
    via the GCS generation precondition. There is no in-memory locking;
    each operation is a fresh round-trip.

    The CLI exposes this behind ``--state-uri``. In step 3 it is wired
    only as a *pure observer* (write-only) inside ``cmd_execute``; the
    orchestrator in step 4 starts using ``read()``.
    """

    DEFAULT_MAX_RETRIES = 5
    DEFAULT_RETRY_BACKOFF_S = 0.5

    def __init__(
        self,
        uri: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        runner: Callable[..., _GsutilResult] = _run_gsutil,
    ):
        validated = parse_state_uri(uri)
        if validated is None:
            raise StateError(
                f"--state-uri must look like gs://bucket/path (got {uri!r})"
            )
        self.uri = validated
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._runner = runner

    # -- public API -----------------------------------------------------

    def read(self, *, required: bool = False) -> Optional[RunState]:
        """Fetch the state object, or None if it does not exist.

        With ``required=True``, raises :class:`StateNotFound` when the
        object is missing.
        """
        payload, _gen = self._read_with_generation()
        if payload is None:
            if required:
                raise StateNotFound(self.uri)
            return None
        return RunState.from_json(payload)

    def write(self, state: RunState) -> None:
        """Unconditional write (overwrites any existing object).

        Used for the very first ``cmd_execute`` write, where there is
        no previous state to read. For incremental updates use
        :meth:`update` so concurrent writers don't clobber each other.
        """
        state.updated_at = _utc_now_iso()
        if not state.created_at:
            state.created_at = state.updated_at
        self._upload(state.to_json(), generation=None)

    def update(self, fn: Callable[[RunState], RunState]) -> RunState:
        """Read, apply ``fn``, write back. Retries on generation conflict.

        ``fn`` may mutate-and-return the same RunState or build a new one.
        Returns the RunState that was successfully written.

        On exhaustion of retries, raises :class:`StateConflict`.
        """
        last_err: Optional[str] = None
        for attempt in range(1, self._max_retries + 1):
            payload, gen = self._read_with_generation()
            if payload is None:
                # Caller is updating before any state was written.
                raise StateNotFound(f"{self.uri} does not exist; call write() first")
            current = RunState.from_json(payload)
            new_state = fn(current)
            new_state.updated_at = _utc_now_iso()
            if not new_state.created_at:
                new_state.created_at = current.created_at or new_state.updated_at
            try:
                self._upload(new_state.to_json(), generation=gen)
                return new_state
            except StateConflict as exc:
                last_err = str(exc)
                # Exponential backoff before the next read-modify-write.
                time.sleep(self._retry_backoff_s * (2 ** (attempt - 1)))
        raise StateConflict(
            f"{self.uri}: {self._max_retries} consecutive write conflicts; "
            f"last error: {last_err}"
        )

    # -- internal helpers ----------------------------------------------

    def _read_with_generation(self) -> tuple[Optional[str], Optional[int]]:
        """Returns (json_payload_or_None, generation_or_None)."""
        # ``gsutil stat`` reports Generation; ``gsutil cat`` returns body.
        stat = self._runner("stat", self.uri)
        if stat.returncode != 0:
            # Treat any non-zero as 'object does not exist'. gsutil prints
            # 'No URLs matched' in that case; other errors bubble up via
            # later operations with clearer messages.
            return None, None
        m = _GEN_RE.search(stat.stdout)
        generation = int(m.group(1)) if m else None

        cat = self._runner("cat", self.uri)
        if cat.returncode != 0:
            return None, None
        return cat.stdout, generation

    def _upload(self, payload: str, *, generation: Optional[int]) -> None:
        """Upload payload as the state object.

        ``generation`` of ``None`` performs an unconditional write.
        ``generation == 0`` precondition means 'object must not exist'.
        Any other integer means 'current generation must match'.
        """
        # Stage to a local tempfile so gsutil cp -h works the same way
        # for stdin and a path. (gsutil reads -h from headers; combining
        # stdin and headers is supported but the file-based path is
        # easier to debug.)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write(payload)
            tmp_path = tf.name
        try:
            args: List[str] = []
            if generation is not None:
                args += ["-h", f"x-goog-if-generation-match:{generation}"]
            args += ["cp", tmp_path, self.uri]
            res = self._runner(*args)
            if res.returncode != 0:
                # 412 Precondition Failed surfaces as a non-zero exit
                # plus 'PreconditionException' or 'condition' in stderr.
                stderr_lower = res.stderr.lower()
                if "precondition" in stderr_lower or "412" in stderr_lower:
                    raise StateConflict(
                        f"generation mismatch on {self.uri}: {res.stderr.strip()}"
                    )
                raise StateError(
                    f"gsutil cp to {self.uri} failed (exit {res.returncode}): "
                    f"{res.stderr.strip()}"
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Convenience constructors used by deploy.py / orchestrator.py
# ---------------------------------------------------------------------------


def initial_state(
    run_id: str,
    config_yaml: str,
    config_hash: str,
) -> RunState:
    """Build a fresh ``RunState`` for the start of a new run."""
    now = _utc_now_iso()
    return RunState(
        run_id=run_id,
        config_yaml=config_yaml,
        config_hash=config_hash,
        schema_version=SCHEMA_VERSION,
        created_at=now,
        updated_at=now,
        phases=Phases(),
    )
