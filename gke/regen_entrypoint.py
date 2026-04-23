#!/usr/bin/env python3
"""
Entrypoint for SpecForge data regeneration on GKE batch jobs.

Pipeline:
  1. Prepare datasets (download from HuggingFace, convert to JSONL).
  2. Launch SGLang server(s) for the target model.
  3. Wait for all servers to become healthy.
  4. Run regenerate_train_data.py for each dataset (chunk-based,
     sharded across pods via --shard-id / --total-shards).
  5. Clean up servers on exit.

Required environment variables:
  JOB_COMPLETION_INDEX  - Set by GKE indexed Job (0-based shard index)
  TOTAL_SHARDS          - Total number of job completions
  MODEL                 - HuggingFace model ID (e.g. google/gemma-4-26b-a4b-it)
  DATASETS              - Comma-separated dataset names for prepare_data.py
                          (or set INPUT_FILES to skip preparation)
  OUTPUT_DIR            - Base directory for output (per-dataset subdirs)

See source for optional environment variables and their defaults.
"""

import atexit
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

SPECFORGE_DIR = Path("/app/specforge")


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------


def env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    """Read an environment variable with optional default."""
    value = os.environ.get(name, default)
    if required and not value:
        print(f"Error: {name} is required.", file=sys.stderr)
        sys.exit(1)
    return value


def env_int(name: str, default: int) -> int:
    return int(env(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env(name, str(default)))


def detect_gpu_count() -> int:
    """Detect number of available GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return len(result.stdout.strip().splitlines()) if result.returncode == 0 else 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0


class Config:
    """All configuration read from environment variables."""

    def __init__(self):
        # Required
        self.job_completion_index = env_int("JOB_COMPLETION_INDEX", -1)
        self.total_shards = env_int("TOTAL_SHARDS", -1)
        self.model = env("MODEL", required=True)
        self.output_dir = env("OUTPUT_DIR", required=True)

        # Datasets / input files (one must be set)
        self.datasets = env("DATASETS", "")
        self.input_files = env("INPUT_FILES", "")
        if not self.datasets and not self.input_files:
            print("Error: Either DATASETS or INPUT_FILES must be set.", file=sys.stderr)
            sys.exit(1)

        # GPU / server
        avail_gpus = detect_gpu_count()
        self.num_servers = env_int("NUM_SERVERS", 1)
        self.tp_size = env_int("TP_SIZE", avail_gpus // max(self.num_servers, 1))
        self.base_port = env_int("BASE_PORT", 30000)
        self.mem_fraction_static = env_float("MEM_FRACTION_STATIC", 0.85)
        self.context_length = env("CONTEXT_LENGTH", "")
        self.extra_sglang_args = env("EXTRA_SGLANG_ARGS", "")
        self.avail_gpus = avail_gpus

        # Regeneration
        self.concurrency = env_int("CONCURRENCY", 128)
        self.max_tokens = env_int("MAX_TOKENS", 2048)
        self.temperature = env_float("TEMPERATURE", 1.0)
        self.is_reasoning_model = env("IS_REASONING_MODEL", "true") == "true"
        self.thinking_ratio = env("THINKING_RATIO", "0.7")
        self.chunk_size = env_int("CHUNK_SIZE", 500)
        self.extra_regen_args = env("EXTRA_REGEN_ARGS", "")

        # Data preparation
        self.prepare_data_dir = env(
            "PREPARE_DATA_DIR",
            os.path.join(self.output_dir, "prepared"),
        )

    def print_config(self):
        print("=" * 60)
        print("  SpecForge Data Regeneration")
        print(f"  Shard {self.job_completion_index} / {self.total_shards}")
        print("=" * 60)
        print(f"  Model:             {self.model}")
        print(f"  TP size:           {self.tp_size}")
        print(f"  Num servers:       {self.num_servers}")
        print(f"  Available GPUs:    {self.avail_gpus}")
        print(f"  Concurrency:       {self.concurrency}")
        print(f"  Max tokens:        {self.max_tokens}")
        print(f"  Temperature:       {self.temperature}")
        print(f"  Reasoning model:   {self.is_reasoning_model}")
        print(f"  Thinking ratio:    {self.thinking_ratio}")
        print(f"  Chunk size:        {self.chunk_size}")
        print(f"  Datasets:          {self.datasets or '<using INPUT_FILES>'}")
        print(f"  Input files:       {self.input_files or '<from prepare_data>'}")
        print(f"  Prepare data dir:  {self.prepare_data_dir}")
        print(f"  Output dir:        {self.output_dir}")
        print(f"  Mem fraction:      {self.mem_fraction_static}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Model prefetch (GCSFuse optimization)
# ---------------------------------------------------------------------------


def prefetch_model(model_path: str) -> Optional[subprocess.Popen]:
    """
    Prefetch model weight files from GCSFuse into page cache.

    GCSFuse is slow for first reads of large files. By reading all
    safetensors files in parallel via dd, we warm the cache before
    SGLang tries to load them. This runs in the background so it
    overlaps with dataset preparation.

    Returns the background process (or None if not a local path).
    """
    if not os.path.isdir(model_path):
        print(f"  Model path {model_path} is not a local directory, skipping prefetch.")
        return None

    # Find all weight files
    weight_files = []
    for root, dirs, files in os.walk(model_path):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt")):
                weight_files.append(os.path.join(root, f))

    if not weight_files:
        print(f"  No weight files found in {model_path}, skipping prefetch.")
        return None

    total_size = sum(os.path.getsize(f) for f in weight_files)
    print(
        f"  Prefetching {len(weight_files)} weight files "
        f"({total_size / 1e9:.1f} GB) from {model_path}..."
    )

    # Build a shell command that reads all files in parallel
    # Each file is read with dd to /dev/null to warm the GCSFuse cache
    dd_cmds = []
    for wf in weight_files:
        dd_cmds.append(
            f'echo "$(date) - Prefetching: {os.path.basename(wf)}" && '
            f'dd if="{wf}" of=/dev/null bs=4M 2>/dev/null'
        )

    # Run all dd commands in parallel using xargs
    script = " & ".join(dd_cmds) + " & wait"
    proc = subprocess.Popen(
        ["bash", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_prefetch(proc: Optional[subprocess.Popen]):
    """Wait for prefetch to complete if still running."""
    if proc is None:
        return
    if proc.poll() is None:
        print("  Waiting for model prefetch to complete...")
        stdout, _ = proc.communicate(timeout=1800)  # 30 min max
        if stdout:
            for line in stdout.decode("utf-8", errors="replace").splitlines()[-5:]:
                print(f"    {line}")
    print("  Model prefetch complete.")


# ---------------------------------------------------------------------------
# Step 1: Prepare datasets
# ---------------------------------------------------------------------------

# Datasets that already exist as JSONL files on GCSFuse.
# These skip prepare_data.py and are used directly.
PREBUILT_DATASETS = {
    "magpie-multilingual": "/gcs/pyc-eagle-data/translate-bp-dataset_train.jsonl",
}


def prepare_datasets(cfg: Config) -> List[str]:
    """
    Run prepare_data.py for each dataset in DATASETS, then append any
    pre-existing files from INPUT_FILES. Returns the combined list of
    input JSONL paths.

    Datasets listed in PREBUILT_DATASETS skip preparation and use the
    pre-existing file directly.

    Only shard 0 runs the preparation; other shards wait for the output
    files to appear on the shared filesystem (GCSFuse).
    """
    input_files = []

    dataset_names = [d.strip() for d in cfg.datasets.split(",") if d.strip()]

    if dataset_names:
        os.makedirs(cfg.prepare_data_dir, exist_ok=True)
        print(f"\nPreparing {len(dataset_names)} dataset(s)...")

        for ds_name in dataset_names:
            # Check if this is a prebuilt dataset (skip prepare)
            if ds_name in PREBUILT_DATASETS:
                path = PREBUILT_DATASETS[ds_name]
                if not os.path.isfile(path):
                    print(
                        f"Error: {ds_name} prebuilt path not found: {path}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print(f"  {ds_name}: using prebuilt file {path}")
                input_files.append(path)
                continue

            # Otherwise, prepare via prepare_data.py
            output_jsonl = os.path.join(cfg.prepare_data_dir, f"{ds_name}_train.jsonl")

            if os.path.isfile(output_jsonl):
                lines = sum(1 for _ in open(output_jsonl))
                print(f"  {ds_name}: already prepared ({lines} lines), skipping.")
            elif cfg.job_completion_index == 0:
                print(f"  {ds_name}: running prepare_data.py...")
                cmd = [
                    sys.executable,
                    str(SPECFORGE_DIR / "scripts" / "prepare_data.py"),
                    "--dataset",
                    ds_name,
                    "--output-path",
                    cfg.prepare_data_dir,
                ]
                print(f"  Running: {shlex.join(cmd)}")
                subprocess.run(cmd, check=True)
            else:
                print(f"  {ds_name}: waiting for shard 0 to prepare data...")
                max_wait = 1800  # 30 minutes
                elapsed = 0
                while not os.path.isfile(output_jsonl) and elapsed < max_wait:
                    time.sleep(10)
                    elapsed += 10
                if not os.path.isfile(output_jsonl):
                    print(
                        f"Error: {output_jsonl} not found after {max_wait}s.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            print(f"  {ds_name}: ready.")
            input_files.append(output_jsonl)

    # Append any explicit INPUT_FILES
    if cfg.input_files:
        extra = [f.strip() for f in cfg.input_files.split(",") if f.strip()]
        input_files.extend(extra)
        print(f"  Added {len(extra)} extra input file(s) from INPUT_FILES.")

    print(f"\nAll input files ({len(input_files)}): {input_files}")
    print("-" * 60)
    return input_files


# ---------------------------------------------------------------------------
# Step 2 & 3: Launch and wait for SGLang servers
# ---------------------------------------------------------------------------


class SGLangServerManager:
    """Manages SGLang server processes with cleanup on exit."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.processes: List[subprocess.Popen] = []
        self.addresses: List[str] = []

    def launch(self):
        """Launch all SGLang server instances."""
        cfg = self.cfg

        for i in range(cfg.num_servers):
            port = cfg.base_port + i * 10
            gpu_start = i * cfg.tp_size
            gpu_end = gpu_start + cfg.tp_size - 1
            cuda_devices = ",".join(str(g) for g in range(gpu_start, gpu_end + 1))

            print(
                f"Starting server {i + 1}/{cfg.num_servers} "
                f"on GPUs {cuda_devices}, port {port}..."
            )

            cmd = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model",
                cfg.model,
                "--tp",
                str(cfg.tp_size),
                "--port",
                str(port),
                "--host",
                "0.0.0.0",
                "--dtype",
                "bfloat16",
                "--cuda-graph-max-bs",
                "128",
                "--trust-remote-code",
                "--mem-fraction-static",
                str(cfg.mem_fraction_static),
            ]

            if cfg.context_length:
                cmd += ["--context-length", cfg.context_length]

            if cfg.extra_sglang_args:
                cmd += shlex.split(cfg.extra_sglang_args)

            env_copy = os.environ.copy()
            env_copy["CUDA_VISIBLE_DEVICES"] = cuda_devices
            # Each server needs a unique torch distributed port to avoid
            # EADDRINUSE when running multiple TP=1 servers on the same node.
            env_copy["MASTER_PORT"] = str(29500 + i)

            log_path = f"/tmp/sglang_server_{port}.log"
            log_file = open(log_path, "w")
            print(f"  Command: {shlex.join(cmd)}")
            print(f"  CUDA_VISIBLE_DEVICES={cuda_devices} MASTER_PORT={29500 + i}")
            print(f"  Log: {log_path}")
            proc = subprocess.Popen(
                cmd,
                env=env_copy,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

            self.processes.append(proc)
            self.addresses.append(f"localhost:{port}")

        # Register cleanup
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown())

    def wait_until_healthy(self, timeout: int = 900):
        """Wait for all servers to respond to health checks."""
        print("\nWaiting for servers to become healthy...")

        for addr in self.addresses:
            url = f"http://{addr}/health"
            elapsed = 0

            while elapsed < timeout:
                # Check if any server process has died
                for proc in self.processes:
                    if proc.poll() is not None:
                        print(
                            f"Error: Server process {proc.pid} died "
                            f"(exit code {proc.returncode})."
                        )
                        self._print_logs()
                        sys.exit(1)

                try:
                    req = urllib.request.Request(url, method="GET")
                    with urllib.request.urlopen(req, timeout=5):
                        pass
                    print(f"  {addr} is healthy. ({elapsed}s)")
                    break
                except (urllib.error.URLError, OSError):
                    if elapsed > 0 and elapsed % 60 == 0:
                        print(f"  {addr} still waiting... ({elapsed}s)")
                    time.sleep(5)
                    elapsed += 5
            else:
                print(f"Error: {addr} did not become healthy within {timeout}s.")
                self._print_logs()
                sys.exit(1)

        print(f"All {len(self.addresses)} server(s) are ready.")
        print("-" * 60)

    def shutdown(self):
        """Gracefully shut down all server processes."""
        if not self.processes:
            return

        print("\nShutting down SGLang server(s)...")

        # SIGTERM first
        for proc in self.processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass

        # Wait briefly then force kill
        time.sleep(3)
        for proc in self.processes:
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

        # Reap
        for proc in self.processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        self.processes.clear()
        print("All servers stopped.")

    def _print_logs(self, tail: int = 100):
        """Print tail of server logs for debugging."""
        for addr in self.addresses:
            port = addr.split(":")[1]
            log_path = f"/tmp/sglang_server_{port}.log"
            if os.path.isfile(log_path):
                with open(log_path) as f:
                    lines = f.readlines()
                total = len(lines)
                print(f"\n--- {log_path} ({total} lines, showing last {tail}) ---")
                for line in lines[-tail:]:
                    print(line, end="")
                print(f"--- end {log_path} ---")
            else:
                print(f"\n--- {log_path}: file not found ---")


# ---------------------------------------------------------------------------
# Step 4: Run regeneration for each dataset
# ---------------------------------------------------------------------------


def run_regeneration(cfg: Config, input_files: List[str], server_addresses: List[str]):
    """Run regenerate_train_data.py for each input file."""
    print(
        f"Starting chunk-based data regeneration for {len(input_files)} dataset(s)...\n"
    )

    for idx, input_file in enumerate(input_files, 1):
        dataset_name = Path(input_file).stem  # e.g. "ultrachat_train"
        dataset_output_dir = os.path.join(cfg.output_dir, dataset_name)

        print("=" * 60)
        print(f"  Dataset {idx}/{len(input_files)}: {dataset_name}")
        print(f"  Input:  {input_file}")
        print(f"  Output: {dataset_output_dir}")
        print("=" * 60)

        if not os.path.isfile(input_file):
            print(f"Warning: {input_file} not found, skipping.")
            continue

        cmd = [
            sys.executable,
            str(SPECFORGE_DIR / "scripts" / "regenerate_train_data.py"),
            "--model",
            cfg.model,
            "--concurrency",
            str(cfg.concurrency),
            "--max-tokens",
            str(cfg.max_tokens),
            "--temperature",
            str(cfg.temperature),
            "--server-address",
            *server_addresses,
            "--input-file-path",
            input_file,
            "--output-dir",
            dataset_output_dir,
            "--chunk-size",
            str(cfg.chunk_size),
            "--shard-id",
            str(cfg.job_completion_index),
            "--total-shards",
            str(cfg.total_shards),
        ]

        if cfg.is_reasoning_model:
            cmd.append("--is-reasoning-model")
            if cfg.thinking_ratio:
                cmd += ["--thinking-ratio", cfg.thinking_ratio]

        if cfg.extra_regen_args:
            cmd += shlex.split(cfg.extra_regen_args)

        print(f"Running: {shlex.join(cmd)}")
        subprocess.run(cmd, check=True)

        print(f"\n  Dataset {dataset_name} complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = Config()
    cfg.print_config()

    # Step 0: Start model prefetch in background (overlaps with data prep)
    print("\nStarting model prefetch...")
    prefetch_proc = prefetch_model(cfg.model)

    # Step 1: Prepare datasets (overlaps with prefetch)
    input_files = prepare_datasets(cfg)

    # Step 2: Launch SGLang servers (prefetch continues in background)
    servers = SGLangServerManager(cfg)
    servers.launch()

    # Step 3: Wait for health (servers load model while prefetch warms cache)
    servers.wait_until_healthy()

    # Step 5: Run regeneration
    run_regeneration(cfg, input_files, servers.addresses)

    print("=" * 60)
    print(f"  Shard {cfg.job_completion_index} complete!")
    print(f"  All {len(input_files)} dataset(s) processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
