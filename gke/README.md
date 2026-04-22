# GKE data regeneration — operator guide

This directory contains the SpecForge data-regeneration pipeline that
runs on GKE. Use this README when you are operating the system. For
design rationale and the rebuild plan, see `gke/PLAN.md`.

> **Status:** mid-rebuild. As of step 4b the entrypoint is
> `python3 gke/orchestrator.py run` (with `run_gke_regen.sh` reduced
> to a thin wrapper). Phases A/B/E run natively; C/D use the helpers
> in `gke.lib.{orchestration,k8s}` directly (legacy candidate-racing,
> until step 6 swaps in strict scheduling). The deploy CLI is now
> only used for `--status` / `--delete` ops shortcuts.
>
> Phase B skips the docker build when the computed `<git-sha>` (or
> `<sha>-dirty<hash>`) tag is already present in Artifact Registry;
> pass `--force-rebuild` to override. The regen YAMLs no longer pin
> a real `image:` tag — the orchestrator writes the resolved tag at
> run time.

## Quick start (current state)

```bash
# Full flow (prepare host-side, then build/deploy/monitor/merge):
python3 gke/orchestrator.py run \
    --config gke/regen-gemma3-27b.yaml \
    --datasets perfectblend,magpie-multilingual \
    --output-dir gs://tcli-llm-test/test-regen-data

# Or, equivalent, via the wrapper:
bash run_gke_regen.sh gke/regen-gemma3-27b.yaml \
    perfectblend,magpie-multilingual \
    gs://tcli-llm-test/test-regen-data

# Resume an existing run by id (state lives in
# gs://<output-bucket>/<run-id>/state.json):
python3 gke/orchestrator.py run \
    --config gke/regen-gemma3-27b.yaml \
    --run-id regen-gemma3-27b-20260422-160000

# Inspect state for a run:
python3 gke/orchestrator.py status \
    --config gke/regen-gemma3-27b.yaml \
    --run-id regen-gemma3-27b-20260422-160000

# Just check live job status across clusters:
python3 gke/deploy.py --yaml gke/regen-gemma3-27b.yaml --status

# Kill all jobs for a config:
python3 gke/deploy.py --yaml gke/regen-gemma3-27b.yaml --delete
```

## Required environment

- `gcloud auth login` and `gcloud config set project cloud-llm-test`.
- `gcloud auth configure-docker us-central1-docker.pkg.dev` (the
  build script does this automatically).
- Docker socket accessible without sudo. If you see
  `permission denied while trying to connect to the Docker daemon
  socket`, run `sudo chmod 666 /var/run/docker.sock` once per session.
  (The auto-fix that used to live in `run_gke_regen.sh` was removed
  in a revert.)
- `kubectl` installed; the deploy script switches contexts via
  `gcloud container clusters get-credentials` automatically.
- For gated HuggingFace models (Gemma): a `hf-token` k8s secret in
  the `default` namespace of every target cluster:
  `kubectl create secret generic hf-token --from-literal=token=hf_...`

## Where things live

| What | Where |
|---|---|
| Container image | `us-central1-docker.pkg.dev/cloud-llm-test/tcli-test/specforge-regen:<tag>` |
| Prepared input data | `gs://pyc-eagle-data/datasets/prepared/<dataset>_train.jsonl` |
| Per-chunk regen output | `gs://<output-bucket>/<run>/<dataset>_train/chunk_NNNNNN.jsonl` + `.done` markers |
| Merged final output | `gs://<output-bucket>/<run>/<dataset>_regen.jsonl` |
| Job state (after orchestrator lands) | `gs://<output-bucket>/<run-id>/state.json` |
| Per-pod sglang server logs | `/tmp/sglang_server_<port>.log` inside container |

## How sharding works

The GKE Job uses `completionMode: Indexed` with `completions: N`. Each
pod reads `JOB_COMPLETION_INDEX` from the downward API and passes it
to `regenerate_train_data.py` as `--shard-id`. The regen script:

1. Splits the input JSONL into chunks of `CHUNK_SIZE` (default 500).
2. Filters to chunks where `chunk_id % total_shards == shard_id`.
3. Skips chunks that already have `chunk_NNNNNN.done` in the output dir.
4. Writes results, then `.done` last (atomic commit).

Because every pod writes a disjoint set of chunk files, there are no
write conflicts. Restarted pods skip done chunks. This is what makes
chunk-level resume work.

The rescue path (Phase D.5, post-step-7) overrides the modulo with an
explicit `--chunk-ids` list to process exactly the chunks that the
primary job missed.

## Reading job status

```bash
# All pods for one dataset's job, with reasons:
kubectl get pods -l job-name=regen-gemma3-27b-perfectblend-h200-europe-west1 \
    -o wide

# Why is a pod Pending?
kubectl describe pod <pod-name> | grep -A20 'Events:'

# Per-shard completion (indexed Job):
kubectl get job regen-gemma3-27b-perfectblend-h200-europe-west1 \
    -o jsonpath='{.status.completedIndexes}'

# Tail logs of one pod:
kubectl logs <pod-name> -f
```

After step 5 lands, the orchestrator's `status` subcommand will surface
all of this with classified pending reasons.

## Common failures

### Pods stuck `Pending` indefinitely

Today this happens silently — `min_running = ceil(N/2)` declares a
winner at half-parallelism, the other pods may stay `Pending` forever
on spot pools at capacity. Symptom: `kubectl get job` shows e.g.
`completions: 4/8` and never advances.

**Workaround until step 6:** check `kubectl describe pod <pending-pod>`
for the `FailedScheduling` event. If the message is
`Insufficient nvidia.com/gpu`, no capacity is available; either delete
and try a different cluster, or wait for autoscaler.

**After step 6:** strict `min_running = total_pods` default; the
orchestrator will fall over to the next candidate within
`attempt_deadline` (10 min default).

**After step 7:** even if shards complete partially, Phase D.5 picks
up the missing chunks via a smaller rescue job.

### `permission denied while trying to connect to the Docker daemon`

`sudo chmod 666 /var/run/docker.sock`. Re-run the command. The
permanent fix is adding your user to the `docker` group at devcontainer
build time.

### `gsutil compose` fails with "max 1024 components"

Fixed in step 4b: `gke.lib.merge.compose_dataset_chunks` recursively
batches at 1000 components per call, composes intermediates into a
final object, and verifies the merged line count against the sum of
`success` counts in the per-chunk `.done` markers. Mismatches are
recorded in state as `merge[ds].status = "unverified"` rather than
raising.

### Pod is `Running` but never produces chunks

Check the sglang server health. Inside the pod:
`tail -f /tmp/sglang_server_30000.log`. Common causes: model weight
download timing out from GCSFuse (look for `Prefetching` lines),
HuggingFace gated-model auth (need `hf-token` secret).

## Per-config knobs (in `gke/regen-*.yaml`)

| Env | Purpose |
|---|---|
| `MODEL` | HF model ID or GCSFuse path |
| `DATASETS` | Comma-separated names matching `prepare_data.py --dataset` choices |
| `OUTPUT_DIR` | `/gcs/<bucket>/<path>` — where chunk files land |
| `CHUNK_SIZE` | Samples per chunk (default 500). Smaller = finer resume; more files |
| `TP_SIZE` | Tensor parallelism per sglang server |
| `NUM_SERVERS` | sglang server instances per pod (each gets `TP_SIZE` GPUs) |
| `CONCURRENCY` | In-flight requests per server |
| `MAX_TOKENS`, `TEMPERATURE` | sampling |
| `IS_REASONING_MODEL`, `THINKING_RATIO` | for reasoning models only |

`TOTAL_SHARDS` is set to match the YAML's `completions:`; do not edit
one without the other.

## Adding a new model

1. Create `configs/<model>-eagle3.json` (training config).
2. Create `gke/regen-<model>.yaml` based on the closest existing one.
   Adjust `MODEL`, `OUTPUT_DIR`, `TP_SIZE`/`NUM_SERVERS` to fit the
   model, and `IS_REASONING_MODEL`/`THINKING_RATIO` to the model class.
3. Run: `bash run_gke_regen.sh gke/regen-<model>.yaml <datasets> <output>`.

## When to read PLAN.md

- You are about to refactor anything in `gke/`. The plan tells you
  which step is in flight and what the target architecture is.
- A behavior in this README seems contradicted by the code. The plan's
  "Step status" section tells you which steps have landed and which
  are still pending — the README may describe target state for a step
  that isn't done yet.
- You want to know why a decision was made. The plan's "Decision log"
  has the answer.
