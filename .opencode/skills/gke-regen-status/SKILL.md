---
name: gke-regen-status
description: Check status of an in-flight or past GKE regen pipeline run (gke/orchestrator.py). Auto-discovers the GCS output bucket from the most recent orchestrator command (running process or /tmp/e2e-run.log), lists all run-ids under it, and surfaces phase progress, pod state, chunk counts, and notable failures from state.json. Use whenever the user asks about an e2e run's status, progress, completion, or failure cause without naming a specific run-id.
license: MIT
compatibility: opencode
metadata:
  audience: orchestrator-operators
  workflow: gke-regen-pipeline
---

# GKE regen pipeline status

Use this skill when the user asks about the **status of a regen pipeline run** — phrases like "how's the e2e going," "what state is the run in," "did it finish," "why did it fail." It works for runs in any state: actively running, completed, failed, or abandoned.

## What this skill does

1. **Auto-discovers the GCS output directory** from one of (in order of preference):
   - `--output-dir` flag in a currently-running `python3 -u gke/orchestrator.py` process (`ps -ef | grep`)
   - The `Output:` line in the orchestrator's banner at the top of `/tmp/e2e-run.log`
   - User-provided override (ask only if both above fail)

2. **Lists all run-ids** under that directory: `gsutil ls <output-dir>/` returns subdirectories named like `regen-<config>-YYYYMMDD-HHMMSS`.

3. **For each run** (or just the most recent if there are many), reports:
   - Whether the orchestrator process is still alive (PID, etime).
   - Phase status: prepare / build / deploy / monitor / rescue / merge — what's done, what's pending.
   - Active Job's pod state if Phase D is in progress (auto-discover cluster + region from `state.deploy[ds].primary_job`).
   - Output chunk count vs expected (state's `prepare[ds].n_lines / chunk_size` from the YAML).
   - Last 10–20 log lines if the orchestrator log is reachable.

## How to invoke

Prefer **calling bash directly** over writing a separate helper script. The work is small enough to inline. The general pattern:

```bash
# 1. Discover the output bucket
OUTPUT_DIR=$(ps -ef | grep -m1 "python3 -u gke/orchestrator" | grep -v grep \
    | grep -oP -- '--output-dir \K[^ ]+' )
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=$(head -20 /tmp/e2e-run.log 2>/dev/null \
        | grep -oP 'Output:\s*\K\S+' | head -1)
fi
[ -z "$OUTPUT_DIR" ] && { echo "Could not auto-discover output dir; ask the user."; exit 0; }
echo "Discovered OUTPUT_DIR=$OUTPUT_DIR"

# 2. List run-ids (newest last because gsutil sorts lexicographically
#    and our run-ids are timestamp-based ``<config>-YYYYMMDD-HHMMSS``).
#    The sed extracts the trailing path component so we don't false-match
#    on bucket prefixes that happen to contain "regen" (e.g. the bucket
#    is named "...-regen-e2e/").
RUNS=$(gsutil ls "${OUTPUT_DIR%/}/" 2>/dev/null | sed -n 's|.*/\([^/]\+\)/$|\1|p')
echo "Found runs:"
echo "$RUNS" | sed 's/^/  /'

# 3. For each (or just the latest), print state summary
LATEST=$(echo "$RUNS" | tail -1)
echo "=== latest: $LATEST ==="
gsutil cat "${OUTPUT_DIR%/}/${LATEST}/state.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  run_id:     {d[\"run_id\"]}')
print(f'  updated_at: {d[\"updated_at\"]}')
for k in ('prepare','build','deploy','monitor','rescue','merge'):
    rec = d['phases'].get(k, {})
    if rec: print(f'  {k:8} {json.dumps(rec)[:200]}')"

# 4. If state.deploy has a primary_job, switch kubectl context and check pods
JOB_INFO=$(gsutil cat "${OUTPUT_DIR%/}/${LATEST}/state.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for ds, rec in d['phases'].get('deploy', {}).items():
    pj = rec.get('primary_job', {})
    if pj.get('job_name'):
        print(f\"{pj['job_name']} {pj['cluster']} {pj['region']}\")")
if [ -n "$JOB_INFO" ]; then
    read JOB CLUSTER REGION <<< "$JOB_INFO"
    gcloud container clusters get-credentials "$CLUSTER" --region "$REGION" \
        --project cloud-llm-test --quiet 2>/dev/null
    echo "=== pods on $CLUSTER ==="
    kubectl get pods -l "job-name=$JOB" --no-headers 2>/dev/null \
        | awk '{print $3}' | sort | uniq -c
fi
```

Inline this pattern, adapt to the user's actual question, surface what's relevant. Don't write the script to disk — keep the work in the conversation.

## When the user wants a specific run

If the user names a run-id (e.g. `regen-gemma3-27b-20260423-210043`), skip step 2 and go straight to inspecting `${OUTPUT_DIR%/}/${run_id}/state.json`. Don't ask for the output dir — derive it the same way.

## When the orchestrator process is alive

The PID is informative (etime tells you how long it's been running, STAT 'S' is normal sleeping). Don't kill it; the user must ask explicitly. If you observe a real stall (e.g. 10+ minutes with no log activity AND no state.json updates), surface it as an observation, not an action.

## When the orchestrator process is gone

That's normal at run completion or after kill. The state.json is still authoritative — read it. The Job in GKE may also still be present; check via the recorded `primary_job` and report its current `kubectl get job` status.

## What this skill does NOT do

- It does not modify state, delete Jobs, or kill processes. Read-only.
- It does not start new runs. The user runs the orchestrator themselves.
- It does not interpret logs beyond surfacing them. Reasoning about *why* a run failed is the agent's job, not the skill's.
- It does not assume the project is `cloud-llm-test`. Read the orchestrator command's `--project` flag the same way as `--output-dir`.

## Notes on the output bucket structure

For reference, an orchestrator run writes to:

```
gs://<bucket>/<output-prefix>/<run-id>/state.json   # the state document
gs://<bucket>/<output-prefix>/<chunk-stem>/         # per-chunk outputs
    chunk_NNNNNN.jsonl                              # actual generated data
    chunk_NNNNNN.done                               # JSON marker per chunk
```

The chunk-stem is `<dataset>_train` for most datasets, `translate-bp-dataset_train` for `magpie-multilingual` (one-off rename in `regen_entrypoint.py`).

To count chunk progress quickly:
```bash
gsutil ls "${OUTPUT_DIR%/}/${chunk_stem}/chunk_*.done" 2>/dev/null | wc -l
```
