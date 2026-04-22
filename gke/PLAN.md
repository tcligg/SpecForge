# GKE regen pipeline v2 — plan

Status tracker for the rebuild of the SpecForge GKE data-regeneration flow.
This file is the source of truth for design and progress.

- **Operational guide:** see `gke/README.md`
- **Workspace agent guidance:** see `/AGENTS.md`
- **Conversation that produced this plan:** OpenCode session on 2026-04-22

## Goals

The current `gke/` workflow has three pain points:

1. **Multi-step, no working one-shot.** `prepare_data.py`, image build, GKE
   submission, monitoring, and merge are run separately. The "one-shot"
   `run_gke_regen.sh` → `gke/deploy.py --execute` flow has bugs and no
   real resume capability.
2. **No idempotency / resume.** Mid-flight failures (docker socket perms,
   spot preemption, partial scheduling) require restarting from scratch.
3. **Partial scheduling silently loses work.** `min_running = ceil(N/2)`
   declares a winner at half-parallelism. If the other half stays
   `Pending`, the indexed Job never reaches `Complete`, the orchestrator
   times out and reports failure, and the chunks owned by stuck shards
   are never processed.

Target after this plan:

- Single command (`gke/orchestrator.py run --config <yaml>`) that drives
  the full flow end-to-end.
- Re-running the command resumes from the last completed step. State
  lives in `gs://<output-bucket>/<run-id>/state.json`.
- Partial scheduling is detected, classified, and either fast-failed
  over to another candidate or recovered via a rescue job that
  processes only the missing chunks.

## Architecture

Six phases, one orchestrator, one state file.

```
gke/orchestrator.py run --config gke/regen-gemma3-27b.yaml [--run-id ID]

  Phase A — PREPARE   (host)   prepare_data.py per dataset → GCS
  Phase B — BUILD     (host)   build_and_push.sh, tag = git-sha
  Phase C — DEPLOY    (GKE)    submit job; wait for FULL parallelism
  Phase D — MONITOR   (GKE)    poll until terminal; track per-shard status
  Phase D.5 — RESCUE  (GKE)    if partial: dispatch shard-aware rescue
  Phase E — MERGE     (host)   recursive gsutil compose to single jsonl
```

Defaults pinned in this plan:

| Knob | Default | Rationale |
|---|---|---|
| `min_running` | `total_pods` (strict) | Prevents partial-completion data loss; configurable down for ops |
| Spot-exhausted strategy | sleep + retry, capped by `--max-wait-hours` | Cheapest; on-demand fallback deferred |
| `prepare_data.py` location | host-side | We run from a GCE workstation; bandwidth is fine |
| `run` blocking | block until done | Simplest; background with `&` if needed |
| State location | `gs://<output-bucket>/<run-id>/state.json` | Co-located with output for ops grep |
| New module layout | `gke/lib/` for refactored helpers | Keeps `gke/` top level focused on the orchestrator |

### State schema (v1)

```jsonc
{
  "run_id": "gemma3-27b-20260422-160000",
  "config_yaml": "gke/regen-gemma3-27b.yaml",
  "config_hash": "sha256:...",
  "schema_version": 1,
  "phases": {
    "prepare": {
      "<dataset>": { "status": "ok", "uri": "gs://...", "size": N, "n_lines": N }
    },
    "build": { "image_uri": "...", "git_sha": "...", "dirty": false },
    "deploy": {
      "<dataset>": {
        "primary_job": { "cluster": "...", "region": "...", "gpu": "...",
                         "job_name": "...", "total_shards": N, "started_at": "..." }
      }
    },
    "monitor": {
      "<dataset>": {
        "primary_job_status": "Complete|Failed|Active",
        "shard_status": { "0": "Complete", "1": "Pending", ... }
      }
    },
    "rescue": {
      "<dataset>": {
        "attempts": [
          { "attempt": 1, "cluster": "...", "gpu": "...", "job_name": "...",
            "covered_chunks": [N, N, ...], "total_shards": N,
            "completed_at": "...", "result": "Complete|Failed" }
        ],
        "all_chunks_done": true
      }
    },
    "merge": {
      "<dataset>": { "uri": "gs://...", "chunks_merged": N, "samples": N }
    }
  }
}
```

## Step-by-step

One PR per step, merged to `eagle3`. Branch naming:
`gke-orchestrator-step-N-<short-name>`.

Mark step status here when landed:

- [x] **Step 0** — docs scaffolding (this file, `gke/README.md`, `AGENTS.md` update)
- [x] **Step 1** — restore chunk-sharding in `scripts/regenerate_train_data.py` + add `--chunk-ids` hook
- [x] **Step 2** — refactor `gke/deploy.py` into `gke/lib/*` + thin CLI shim
- [x] **Step 3** — add `gke/state.py` as no-op observer in current `--execute` flow
- [x] **Step 4a** — `gke/orchestrator.py` scaffold + Phase A (prepare); `run_gke_regen.sh` reduced to wrapper
- [x] **Step 4b** — Phase B (build, with GAR-cache skip) + Phase E (recursive compose, line-count verify); regen YAMLs unpinned; orchestrator no longer shells out to `deploy.py --execute`
- [ ] **Step 5** — `gke/scheduling.py` (pending-reason classifier) behind feature flag
- [ ] **Step 6** — move Phases C/D into orchestrator with strict scheduling
- [ ] **Step 7** — `gke/rescue.py` and Phase D.5
- [ ] **Step 8** — cleanup (delete `.sh` duplicates, deprecate `deploy.py` CLI, sync YAMLs)

### Step 1 — Restore chunk-sharding

**Files:** `scripts/regenerate_train_data.py`

**Why first:** the file as committed cannot run. `main()` references
undefined `pending_chunks`/`already_done` and legacy
`args.output_file_path`. The shard-aware planning code existed in
earlier commits (verified via `git log -p`) and was lost in a revert.
Every later step assumes a working regen script.

**Changes:**
- Replace lines ~499–521 with chunked planning:
  - Read input, split into chunks of `args.chunk_size`.
  - `my_chunks = [(cid, lines) for cid, lines in enumerate(all_chunks) if cid % args.total_shards == args.shard_id]`
  - `pending_chunks = [(cid, lines) for cid, lines in my_chunks if not is_chunk_done(args.output_dir, cid)]`
- Add `--chunk-ids` arg (comma-separated explicit chunk IDs). When set,
  override modulo: shard `i` of `total_shards` processes
  `chunk_ids[j]` for `j % total_shards == i`. **Rescue-job hook**
  added now to avoid touching this file again in step 7.
- Validate `0 <= shard_id < total_shards`, `chunk_size > 0`.
- Delete `--output-file-path` and `--resume` (already marked deprecated).

**Verification:**
- Local smoke: tiny JSONL, `--total-shards 2 --shard-id 0/1` produce
  disjoint `chunk_NNNNNN.done` covering all chunks.
- `--chunk-ids 0,3,5 --total-shards 2 --shard-id 0` processes 0 and 5.
- Re-run is no-op.

### Step 2 — Refactor deploy.py into library

**Files:**
- `gke/lib/__init__.py`
- `gke/lib/clusters.py` — `discover_clusters`, `use_cluster`
- `gke/lib/k8s.py` — `get_pod_status`, `delete_job`, `run_output`, `_wait_for_job_completion`
- `gke/lib/yaml_patch.py` — `patch_and_apply_yaml`, `make_job_name`, `gpu_short_name`
- `gke/lib/build.py` — `_build_and_push_image`
- `gke/lib/merge.py` — `_merge_output_chunks`
- `gke/deploy.py` — becomes thin CLI shim importing from `gke.lib`

**Constraint:** mechanical extraction only. **No logic edits.**

**Verification:** `gke/deploy.py --status` and `--delete` produce
identical output to before. `--execute` skipped (orthogonal docker
issue).

### Step 3 — State module as observer

**Files:**
- `gke/state.py` — new
- `gke/deploy.py` — minimal hooks behind `--state-uri` flag

**`gke/state.py` API:**
- `class RunState` — dataclass mirroring schema above.
- `class StateStore`:
  - `read() -> RunState | None`
  - `write(state)` — atomic via `If-Generation-Match`
  - `update(fn)` — read-modify-write with bounded retry on conflict
- `compute_run_id(config_path, prefix)`
- `compute_config_hash(config_path)` — sha256 of file
- Implementation: `gsutil cp` with `-h x-goog-if-generation-match:N`
  (no new Python deps).

**Hooks:** `--state-uri gs://...` flag in `gke/deploy.py`, when set,
writes state at each phase boundary in `cmd_execute`. Reads no state.
Pure observer.

**Verification:**
- Round-trip a `RunState` against a real GCS bucket.
- Run current `--execute` with `--state-uri`, inspect resulting JSON.
- Concurrent `update()` test: second writer retries cleanly.

**Manual smoke against real GCS** (run once after step 3 lands; not in CI):

```bash
# Round-trip + concurrent-update check.
URI=gs://tcli-llm-test/test-state/$(date +%s).json
python3 - <<'PY'
import os
from gke import state
uri = os.environ["URI"]
store = state.StateStore(uri)
s = state.initial_state("smoke-1", "gke/regen-gemma3-27b.yaml", "sha256:0")
store.write(s)
loaded = store.read()
assert loaded.run_id == "smoke-1"

def add_image(s):
    s.phases.build = state.BuildPhase(image_uri="img:1", git_sha="abc",
                                       dirty=False, completed_at="now")
    return s
store.update(add_image)
print("round-trip OK; final state:")
print(store.read().to_json())
PY
gsutil rm "$URI"
```

### Step 4a — Orchestrator scaffold + Phase A

**Files:**
- `gke/orchestrator.py` — new (scaffold + Phase A only)
- `run_gke_regen.sh` — replaced by 5-line wrapper

**Scaffold contents:**
```
class OrchestratorConfig
def cmd_run(cfg)
def cmd_status(cfg)
def cmd_logs(cfg)        # stub
def cmd_cancel(cfg)      # stub
def cmd_cleanup(cfg)     # stub
def cmd_rescue(cfg)      # stub
def phase_a_prepare(cfg, state)
def phase_b_build(cfg, state)    # stub: shells out to existing flow
def phase_c_deploy(cfg, state)   # stub: shells out to deploy.py
def phase_d_monitor(cfg, state)  # stub: shells out to deploy.py
def phase_e_merge(cfg, state)    # stub: shells out to deploy.py
def main()
```

**Phase A logic:**
- For each `cfg.datasets`:
  - Skip if `state.phases.prepare[ds].status == "ok"` and GCS file
    exists with size ≥ recorded.
  - Else: `subprocess.run(["python3", "scripts/prepare_data.py", ...])`,
    `gsutil cp` to GCS, `state.update()`.
- Record line count for Phase D.5 use.

**Re-run flags:** `--run-id`, `--new`, `--from <phase>`,
`--ignore-config-change`, `--force-rebuild` — wired to argparse, only
`--from` and `--new` functional in this step.

**Verification:** end-to-end run still works (other phases
delegated). Re-running skips Phase A.

### Step 4b — Phases B and E

**Files:** `gke/orchestrator.py` extended.

**Phase B (build):**
- Tag = `<git-sha>` if clean, `<git-sha>-dirty<sha256(diff)[:8]>` if dirty.
- Skip if `gcloud artifacts docker images describe <full-tag>` succeeds.
- Else shell out to `gke/build_and_push.sh` with `TAG=<computed>`.
- Record `image_uri` in state.

**Phase E (merge):**
- Use `gke.lib.merge._merge_output_chunks`, extended with **recursive
  compose**: if `len(chunks) > 1024`, compose in batches of 1024 into
  intermediate objects, then compose intermediates. Verify final line
  count matches `sum(stats["success"])` from chunk `.done` markers.

YAML changes also land here:
- `gke/regen-gemma3-27b.yaml` and `gke/regen-gemma4-26b.yaml`: stop
  pinning `image:` (orchestrator writes it). Stop relying on
  `DATASETS` env for in-container prepare.

### Step 5 — Pending-reason classifier

**Files:**
- `gke/scheduling.py` — new
- `gke/lib/k8s.py` — extended
- `gke/deploy.py` — wire behind `--enable-scheduling-v2` flag

**Classes:**
```
class PendingReason(Enum):
    CAPACITY_WAIT          # autoscaler scaling, retry
    CAPACITY_EXHAUSTED     # no capacity, no scale-up event
    CONFIG_ERROR           # affinity / volume / secret
    IMAGE_PULL_ERROR
    UNKNOWN

@dataclass
class PodDiagnosis:
    pod_name: str
    phase: str
    reason: PendingReason | None
    raw_message: str
    last_event_age_seconds: int | None

def diagnose_job(job_name) -> list[PodDiagnosis]
```

**Decision matrix (per pod):**

| Class | Signal | Deploy action |
|---|---|---|
| `CAPACITY_WAIT` | `Insufficient nvidia.com/gpu` AND `TriggeredScaleUp` event in last 5 min | keep waiting |
| `CAPACITY_EXHAUSTED` | `Insufficient nvidia.com/gpu` AND no scale-up AND deadline passed | next candidate |
| `CONFIG_ERROR` | volume/secret/affinity error | bail dataset |
| `IMAGE_PULL_ERROR` | `ImagePullBackOff`, `ErrImagePull` | bail dataset |
| `UNKNOWN` | anything else | treat as `CAPACITY_WAIT` |

**Default: flag off.** Step 6 turns it on as default.

### Step 6 — Phases C/D in orchestrator + strict scheduling

**Files:** `gke/orchestrator.py` extended. `gke/deploy.py` adds
deprecation warning to `--execute`.

**Defaults:**
- `min_running = total_pods` (strict)
- `attempt_deadline = 600` (10 min per candidate)
- `monitor_deadline = 14400` (4h after last shard transition)
- `spot_exhausted_strategy = "sleep_retry"`
- `spot_retry_interval = 900`
- `max_wait_hours = 12`

**Phase C algorithm:**
```
candidates = order_candidates(cfg, ds)
for cluster, region, gpu in candidates:
    submit job
    result = wait_for_full_scheduling(job, target=min_running, deadline=attempt_deadline)
    match result:
        WIN: record, return
        CONFIG_ERROR | IMAGE_PULL_ERROR: delete, raise DeployError
        CAPACITY_EXHAUSTED | TIMEOUT: delete, continue
all candidates exhausted:
    if spot_exhausted_strategy == "sleep_retry": sleep, recurse (bounded)
    else: raise
```

**Adopt-existing-job:** before submitting, check if `state.deploy[ds]`
points to a Job that still exists and is Active/Complete. If so,
skip submission.

### Step 7 — Rescue (Phase D.5)

**Files:**
- `gke/rescue.py` — new
- `gke/orchestrator.py` — wire D.5 between D and E
- `gke/regen_entrypoint.py` — read `CHUNK_IDS` env, pass to regen via
  `--chunk-ids` (script-side already supports this from step 1)

**Trigger conditions (any):**
- `state.monitor[ds].primary_job_status == "Failed"`
- `state.monitor[ds].primary_job_status == "Active"` past `monitor_deadline`
- Mixed shard status (some Complete, some Pending past 1h)

**Algorithm:**
```
for attempt in range(1, max_rescue_attempts + 1):
    missing = compute_missing_chunks(state, ds)
    if not missing: state.update_rescue(all_chunks_done=True); return
    rescue_shards = min(len(missing), max_rescue_shards)
    submit_rescue_job(ds, attempt, rescue_shards, missing, candidates)
    monitor_result = monitor_job(...)
    state.append_rescue_attempt(...)
final missing != 0 → RescueError
```

**Subcommand:** `gke/orchestrator.py rescue --run-id ID --dataset NAME`
for manual trigger.

**Defaults:** `max_rescue_attempts = 3`, `max_rescue_shards = 4`.

### Step 8 — Cleanup

- Delete `gke/deploy.sh` (700-line dead bash).
- Delete `gke/regen_entrypoint.sh` (dead bash duplicate).
- Reduce `gke/deploy.py` to deprecation shim (keep `--status`, `--delete`
  as ops shortcuts; redirect `--execute` to orchestrator).
- Remove deleted `.sh` files from `gke/Dockerfile` `COPY`.
- Sync `gke/regen-gemma4-26b.yaml` to new GAR path
  (`cloud-llm-test/tcli-test`).
- Update `AGENTS.md` and `gke/README.md` to reflect final state.

## File map after completion

```
gke/
├── PLAN.md                  (this file, kept as historical record)
├── README.md                (operator guide)
├── orchestrator.py          (single CLI entrypoint)
├── state.py                 (GCS-backed state machine)
├── scheduling.py            (pending-reason classifier)
├── rescue.py                (Phase D.5)
├── deploy.py                (deprecation shim, --status / --delete only)
├── lib/
│   ├── __init__.py
│   ├── clusters.py
│   ├── k8s.py
│   ├── yaml_patch.py
│   ├── build.py
│   └── merge.py
├── build_and_push.sh        (unchanged)
├── Dockerfile               (no longer copies deleted .sh)
├── regen_entrypoint.py      (drops shard-0-prepare logic; reads CHUNK_IDS)
├── regen-gemma3-27b.yaml    (image: unpinned)
└── regen-gemma4-26b.yaml    (image: unpinned, GAR path synced)
```

## Open items deferred

- Replace string-based YAML patching with `ruamel.yaml`. After step 7,
  surface area is small (~3 fields); urgency dropped.
- Per-cluster historical success-rate ordering of candidates. Could
  read aggregates from past `state.json` files. Not needed for v1.
- Pin `transformers` in `pyproject.toml` instead of `pip install -U`
  in Dockerfile. Independent of this work.
- On-demand GPU fallback when spot is globally exhausted. Currently
  sleep + retry; revisit if `--max-wait-hours` is regularly hit.
- `--chunk-ids` env var size: if missing-chunk lists exceed kubectl
  env-var size limits (~1MB practical), write list to GCS and pass
  URI. Defer until observed.

## Decision log (for future-me)

| Date | Decision | Why |
|---|---|---|
| 2026-04-22 | One PR per step, merged to `eagle3` | Reviewable, revertable |
| 2026-04-22 | PLAN.md only, no GitHub umbrella issue | Solo work on fork |
| 2026-04-22 | `gke/lib/` subdirectory for refactored helpers | Keeps `gke/` top level focused |
| 2026-04-22 | Step 4 split into 4a + 4b | Scaffold separately from Phase B/E logic |
| 2026-04-22 | `min_running = total_pods` strict default | Phase D.5 makes partial recovery safe; eliminates silent data loss |
| 2026-04-22 | Sleep-retry on spot exhausted (not on-demand fallback) | Cheapest; revisit if hit often |
| 2026-04-22 | `prepare_data.py` runs host-side | GCE workstation has bandwidth |
| 2026-04-22 | `run` blocks until completion | Simplest UX; background with `&` |
| 2026-04-22 | Steps 1+2 land on the same branch / PR | Iterating on a working branch toward `eagle3`; not actually opening per-step PRs |
| 2026-04-22 | `Config` lives in `gke/lib/config.py`, not `gke/deploy.py` | Avoids circular import (deploy → lib → Config). Small departure from the step 2 spec; mechanical extraction otherwise |
| 2026-04-22 | Drop underscore prefixes on extracted lib functions; `gke/deploy.py` re-exports old names as aliases | Cleaner public API for orchestrator; back-compat for any external `from gke.deploy import _foo` users |
| 2026-04-22 | Add `gke/__init__.py` and exclude `gke*` from `pyproject.toml`'s `packages.find` | Explicit package (PEP 420 namespace packages misbehave with some tools); excluded so the `specforge` wheel does not ship operator tooling |
| 2026-04-22 | State schema: top-level `RunState` dataclassed; per-dataset entries as plain dicts | Pragmatic balance — type safety where it pays (the always-present envelope), open-ended where shapes are still evolving (per-dataset records will gain fields in steps 6/7) |
| 2026-04-22 | All GCS I/O via `gsutil` CLI subprocess (no `google-cloud-storage` dep) | Per PLAN's "no new Python deps" constraint; rest of the codebase already shells out to gsutil |
| 2026-04-22 | Step 4a: `run` always starts a fresh run unless `--run-id` given | Auto-discovery (latest state.json by config_hash) is more code and more failure modes; defer until step 6. `--new` is accepted but redundant in this step (kept for forward compat) |
| 2026-04-22 | Step 4a: state URI auto-derived from YAML `OUTPUT_DIR` mount | YAML already encodes the bucket via `OUTPUT_DIR: /gcs/<bucket>/...`. Mapping to `gs://<bucket>/.../<run_id>/state.json` matches PLAN's "co-located with output" rule and avoids a redundant flag on the common path. `--state-uri` overrides for ops |
| 2026-04-22 | Step 4a: orchestrator delegates Phases B/C/D/E to `gke/deploy.py --execute` as one unit | Smallest landable change. Each native phase replaces its slice of the delegated call in steps 4b (build/merge) and 6 (deploy/monitor). Threads `--state-uri` so the step-3 observer keeps populating state |
| 2026-04-22 | Separate `OrchestratorConfig` (in `gke/orchestrator.py`) from `gke.lib.config.Config` | Different inputs (orchestrator owns run identity, state URI, resume flags). When B/C/D/E go native the deploy `Config` shrinks; keeping them split now avoids a churnful merge later |
| 2026-04-22 | Step 4a: stub subcommands (`logs`, `cancel`, `cleanup`, `rescue`) hard-exit with a pointer | Reserves the CLI surface so users see "this is coming" instead of `unknown command`, without committing to behavior we haven't designed |
| 2026-04-22 | Step 4b: orchestrator stops shelling out to `gke/deploy.py --execute`; Phases C/D call `gke.lib.{orchestration,k8s}` helpers directly | Avoids the awkward "delegate but skip rebuild and remerge" dance once Phase B/E are native. `deploy.py` shrinks toward its eventual step-8 deprecation shim (`--status` / `--delete` only). The legacy candidate-racing logic in `gke.lib.orchestration` stays intact behind the helper for now; step 6 replaces it |
| 2026-04-22 | Step 4b: gsutil compose batch capped at 1000 (not 1024); recursion depth capped at 4 | 1000 leaves headroom against the 1024 service limit, which has bitten compose pipelines that hit it on the boundary. Depth 4 with batch 1000 covers up to 10^12 source objects — orders of magnitude past anything realistic, but high enough that an off-by-one in the batching math fails fast instead of looping |
| 2026-04-22 | Step 4b: line-count verifier records mismatch in state instead of raising | Mismatch usually means a few chunks were skipped/missing rather than corrupt data; killing the whole run forces the operator to re-run everything. Recording `status="unverified"` plus the expected/actual gap lets `status` surface it and an operator decide |
| 2026-04-22 | Step 4b: regen YAMLs unpinned to `:UNPINNED` placeholder + comment | Phase B writes the resolved tag at run time; the placeholder keeps the file kubectl-applyable when debugging by hand. Tightened the `image:` substitution regex in both `gke/orchestrator.py` and `gke/deploy.py` so the explanatory comment doesn't get rewritten instead of the real key |
| 2026-04-22 | Step 4b: Phase B "dirty" diff ignores untracked files | The workspace routinely accumulates local artifacts (sample jsonl, results/) that aren't part of the build context. Hashing only `git diff HEAD` keeps the tag stable across runs and avoids spurious "dirty" rebuilds. PLAN.md's spec is satisfied in spirit (clean = `<sha>`, modified = `<sha>-dirty<hash>`) |
| 2026-04-22 | Step 4b: `--force-rebuild` salts the tag with wall-clock, not just bypasses GAR-describe | If we only bypassed describe and pushed to the same tag, GAR would silently keep the older image (Artifact Registry treats tag pushes as moves but the existing image remains pullable until GC). Salting guarantees a brand-new tag the YAML can pin |
