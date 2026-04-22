"""Chunk merging via gsutil compose.

Step 4b of gke/PLAN.md extends this module with recursive compose
(unbounded chunk count) and a line-count verifier driven by the
``chunk_NNNNNN.done`` markers that ``scripts/regenerate_train_data.py``
writes alongside each chunk.

Backwards compatibility: ``merge_output_chunks(cfg, dataset)`` keeps
the same signature deploy.py uses today, so the step-3 observer in
``cmd_execute`` continues to work without change. New callers
(orchestrator's Phase E) should prefer ``compose_dataset_chunks``,
which returns a structured result for state recording.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from gke.lib.config import Config
from gke.lib.k8s import run_output

# gsutil compose hard cap. We deliberately leave headroom (1000 vs 1024)
# so an off-by-one on a partial level doesn't blow up the whole job.
_GSUTIL_COMPOSE_BATCH = 1000

# Cap recursion depth to catch infinite-loop bugs in compose math. With
# batch=1000 this still permits 1000^4 = 10^12 source objects; far past
# anything a real run will see.
_MAX_RECURSION_DEPTH = 4


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MergeResult:
    """Outcome of merging one dataset's chunk files into a single object."""

    dataset: str
    merged_uri: str
    chunks_merged: int
    intermediate_objects: int = 0  # how many temp objects compose created
    expected_lines: Optional[int] = None  # sum of "success" across .done files
    actual_lines: Optional[int] = None  # `gsutil cat | wc -l`-equivalent
    skipped_chunks: List[str] = field(default_factory=list)  # missing .done

    @property
    def verified(self) -> bool:
        """True when expected & actual line counts agree (and were computed)."""
        return (
            self.expected_lines is not None
            and self.actual_lines is not None
            and self.expected_lines == self.actual_lines
        )

    def to_state_dict(self) -> dict:
        """Shape that lands in ``state.phases.merge[dataset]``."""
        return {
            "uri": self.merged_uri,
            "chunks_merged": self.chunks_merged,
            "intermediate_objects": self.intermediate_objects,
            "expected_lines": self.expected_lines,
            "actual_lines": self.actual_lines,
            "verified": self.verified,
            "skipped_chunk_count": len(self.skipped_chunks),
        }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_output_dir(cfg: Config) -> Optional[str]:
    """Return the gs:// base URI for chunk output, or None on failure.

    Mirrors the legacy logic: ``cfg.output_dir`` wins; otherwise parse
    ``OUTPUT_DIR: "/gcs/<bucket>/<path>"`` out of the YAML.
    """
    if cfg.output_dir:
        return cfg.output_dir
    with open(cfg.job_yaml) as f:
        yaml_content = f.read()
    match = re.search(r'name: OUTPUT_DIR\s+value: "/gcs/([^"]+)"', yaml_content)
    if not match:
        print(
            f"Warning: Could not determine OUTPUT_DIR from "
            f"{cfg.job_yaml}. Cannot merge."
        )
        return None
    return "gs://" + match.group(1)


def _dataset_chunk_stem(dataset: str) -> str:
    """Map a dataset name to the per-chunk subdirectory stem.

    Mirrors ``regen_entrypoint.py``: the entrypoint computes
    ``Path(input_file).stem`` and writes chunks into
    ``<OUTPUT_DIR>/<stem>/chunk_*.jsonl``. For the ``magpie-multilingual``
    prebuilt dataset the input file is ``translate-bp-dataset_train.jsonl``,
    so the stem differs from the dataset name. All other datasets use
    ``<dataset>_train``.
    """
    if dataset == "magpie-multilingual":
        return "translate-bp-dataset_train"
    return f"{dataset}_train"


def chunks_dir_uri(cfg: Config, dataset: str) -> Optional[str]:
    base = _resolve_output_dir(cfg)
    if base is None:
        return None
    base = base.rstrip("/")
    return f"{base}/{_dataset_chunk_stem(dataset)}"


def merged_file_uri(cfg: Config, dataset: str) -> Optional[str]:
    base = _resolve_output_dir(cfg)
    if base is None:
        return None
    base = base.rstrip("/")
    return f"{base}/{dataset}_regen.jsonl"


# ---------------------------------------------------------------------------
# gsutil compose primitives
# ---------------------------------------------------------------------------


def _gsutil_ls(uri_pattern: str) -> List[str]:
    out = run_output(["gsutil", "ls", uri_pattern])
    if not out:
        return []
    return [line for line in out.splitlines() if line.strip()]


def _gsutil_rm(uri: str, *, ignore_missing: bool = True) -> None:
    proc = subprocess.run(
        ["gsutil", "rm", uri],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 and not ignore_missing:
        raise RuntimeError(f"gsutil rm {uri} failed: {proc.stderr.strip()}")


def _gsutil_compose(sources: List[str], dest: str) -> None:
    """Single ``gsutil -m compose`` call. Caller must ensure len <= 1000."""
    if len(sources) > _GSUTIL_COMPOSE_BATCH:
        raise ValueError(
            f"_gsutil_compose got {len(sources)} sources; "
            f"cap is {_GSUTIL_COMPOSE_BATCH}. Use _recursive_compose."
        )
    cmd = ["gsutil", "-m", "compose"] + sources + [dest]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gsutil compose -> {dest} failed: {proc.stderr.strip()}")


def _intermediate_uri(dest: str, level: int, batch_idx: int) -> str:
    """Synthesize a deterministic temp object name next to ``dest``.

    Lives under a hidden prefix so manual ``gsutil ls`` of the merged
    file's directory doesn't surface them; the cleanup pass deletes all
    of them after the final compose succeeds.
    """
    suffix = f".compose-tmp.L{level}.b{batch_idx:05d}"
    return f"{dest}{suffix}"


def _recursive_compose(sources: List[str], dest: str) -> int:
    """Compose ``sources`` into ``dest``, batching when above the limit.

    Returns the count of intermediate objects written (and subsequently
    deleted). Caller is responsible for ensuring ``dest`` does not exist;
    this function will not attempt to overwrite.
    """
    if not sources:
        raise ValueError("Cannot compose an empty source list.")

    intermediates_total = 0
    level = 0
    current = list(sources)
    intermediates_to_clean: List[str] = []

    while len(current) > _GSUTIL_COMPOSE_BATCH:
        if level >= _MAX_RECURSION_DEPTH:
            raise RuntimeError(
                f"Compose recursion exceeded depth {_MAX_RECURSION_DEPTH}; "
                f"refusing to continue with {len(current)} objects."
            )
        next_level: List[str] = []
        for batch_idx in range(0, len(current), _GSUTIL_COMPOSE_BATCH):
            batch = current[batch_idx : batch_idx + _GSUTIL_COMPOSE_BATCH]
            inter = _intermediate_uri(dest, level, batch_idx // _GSUTIL_COMPOSE_BATCH)
            # Make sure no leftover from a previous attempt blocks compose.
            _gsutil_rm(inter, ignore_missing=True)
            _gsutil_compose(batch, inter)
            next_level.append(inter)
            intermediates_to_clean.append(inter)
            intermediates_total += 1
        current = next_level
        level += 1

    # Final compose into dest.
    _gsutil_compose(current, dest)

    # Best-effort cleanup. Don't fail the merge over leftover temps;
    # state still records the count so an operator can sweep manually.
    for tmp in intermediates_to_clean:
        _gsutil_rm(tmp, ignore_missing=True)

    return intermediates_total


# ---------------------------------------------------------------------------
# .done marker accounting
# ---------------------------------------------------------------------------


_CHUNK_FILE_RE = re.compile(r".*/chunk_(\d{6})\.jsonl$")
_DONE_FILE_RE = re.compile(r".*/chunk_(\d{6})\.done$")


def _chunk_id(uri: str) -> Optional[int]:
    m = _CHUNK_FILE_RE.match(uri)
    return int(m.group(1)) if m else None


def _done_id(uri: str) -> Optional[int]:
    m = _DONE_FILE_RE.match(uri)
    return int(m.group(1)) if m else None


def _read_done_marker(uri: str) -> Optional[dict]:
    """Read a .done marker (small JSON blob) via ``gsutil cat``."""
    proc = subprocess.run(
        ["gsutil", "cat", uri],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _expected_lines_from_done(
    chunks_dir: str, chunk_ids: Iterable[int]
) -> Tuple[Optional[int], List[str]]:
    """Sum the ``success`` counts across the listed chunks' .done files.

    Returns ``(total, missing)`` where ``missing`` lists chunk URIs whose
    .done marker could not be read (treated as zero contributors). If
    every marker is missing, returns ``(None, missing)`` — caller can
    decide whether to skip verification or fail.
    """
    chunk_ids = list(chunk_ids)
    if not chunk_ids:
        return 0, []

    missing: List[str] = []
    total = 0
    seen_any = False
    for cid in chunk_ids:
        done_uri = f"{chunks_dir}/chunk_{cid:06d}.done"
        marker = _read_done_marker(done_uri)
        if marker is None:
            missing.append(done_uri)
            continue
        seen_any = True
        total += int(marker.get("success", 0))
    if not seen_any:
        return None, missing
    return total, missing


def _count_lines_in_object(uri: str) -> Optional[int]:
    """Count newlines in a GCS object via ``gsutil cat | wc -l``.

    Returns None on subprocess failure. Uses a shell pipeline so we
    don't have to slurp the full file into Python memory.
    """
    proc = subprocess.run(
        f"gsutil cat {uri} | wc -l",
        shell=True,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compose_dataset_chunks(
    cfg: Config,
    dataset: str,
    *,
    verify_lines: bool = True,
    overwrite: bool = True,
) -> Optional[MergeResult]:
    """Merge all ``chunk_NNNNNN.jsonl`` for ``dataset`` into one file.

    Returns a ``MergeResult`` (or None when there is nothing to merge or
    the output directory cannot be resolved). When ``verify_lines`` is
    set we additionally read each chunk's ``.done`` marker, sum the
    ``success`` counts, and compare against the merged object's line
    count. The verification result is recorded on the result dataclass
    rather than raised, so callers (orchestrator, deploy CLI) can decide
    whether a mismatch is fatal.
    """
    chunks_dir = chunks_dir_uri(cfg, dataset)
    merged_uri = merged_file_uri(cfg, dataset)
    if chunks_dir is None or merged_uri is None:
        return None

    chunk_uris = _gsutil_ls(f"{chunks_dir}/chunk_*.jsonl")
    if not chunk_uris:
        print(f"Warning: No chunk files found at {chunks_dir}. Nothing to merge.")
        return None

    chunk_ids: List[int] = []
    sorted_uris: List[str] = []
    for uri in chunk_uris:
        cid = _chunk_id(uri)
        if cid is None:
            # Defensive: gsutil ls shouldn't return non-matching paths,
            # but if it does we keep the URI in deterministic order so
            # compose behaviour stays reproducible. Skip from .done math.
            sorted_uris.append(uri)
            continue
        chunk_ids.append(cid)
    # Re-sort with chunk_id as the key when available; pure-string
    # sorting works because of zero-padded ids, but we prefer to be
    # explicit so a future filename change doesn't silently shuffle.
    sorted_uris = sorted(chunk_uris, key=lambda u: (_chunk_id(u) or 1 << 30, u))
    chunk_ids = sorted({_chunk_id(u) for u in sorted_uris if _chunk_id(u) is not None})

    print(f"  Merging {len(sorted_uris)} chunks for {dataset} -> {merged_uri}")

    # Make sure compose has a clean target. ``compose`` won't overwrite
    # an existing object; the legacy helper just fired ``gsutil rm``
    # before composing. We do the same, optionally bypassed.
    if overwrite:
        _gsutil_rm(merged_uri, ignore_missing=True)
    elif _gsutil_ls(merged_uri):
        raise RuntimeError(
            f"Refusing to overwrite existing {merged_uri} (pass overwrite=True)."
        )

    t0 = time.time()
    intermediates = _recursive_compose(sorted_uris, merged_uri)
    elapsed = time.time() - t0
    print(
        f"  Composed {len(sorted_uris)} chunks "
        f"({intermediates} intermediates) in {elapsed:.1f}s."
    )

    expected = actual = None
    skipped: List[str] = []
    if verify_lines:
        expected, missing = _expected_lines_from_done(chunks_dir, chunk_ids)
        skipped = missing
        if expected is None:
            print(
                f"  Warning: no .done markers readable; skipping line-count "
                f"verification for {dataset}."
            )
        else:
            actual = _count_lines_in_object(merged_uri)
            if actual is None:
                print(
                    f"  Warning: could not count lines in {merged_uri}; "
                    f"verification skipped."
                )
            elif actual != expected:
                print(
                    f"  WARNING: merged line count mismatch for {dataset}: "
                    f"expected {expected} (sum of .done success), "
                    f"got {actual}."
                )
            else:
                print(f"  Verified: {actual} lines == sum of .done success counts.")

    return MergeResult(
        dataset=dataset,
        merged_uri=merged_uri,
        chunks_merged=len(sorted_uris),
        intermediate_objects=intermediates,
        expected_lines=expected,
        actual_lines=actual,
        skipped_chunks=skipped,
    )


def merge_output_chunks(cfg: Config, dataset: str):
    """Legacy entrypoint preserved for ``gke/deploy.py``.

    Delegates to ``compose_dataset_chunks`` with verification enabled.
    Step-4b note: callers that need the structured result (orchestrator
    Phase E) should call ``compose_dataset_chunks`` directly.
    """
    try:
        result = compose_dataset_chunks(cfg, dataset)
    except RuntimeError as exc:
        print(f"Error: gsutil compose failed for {dataset}: {exc}")
        return
    if result is None:
        return
    if not result.verified and result.expected_lines is not None:
        print(
            f"Warning: {dataset} merged but did not verify "
            f"(expected={result.expected_lines}, actual={result.actual_lines})."
        )
    else:
        print(f"  Successfully merged {result.chunks_merged} chunks for {dataset}.")
