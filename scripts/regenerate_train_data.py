"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model's output distribution.

Usage:
1. Set up one or more SGLang servers for the target model.

python3 -m sglang.launch_server \
	--model Qwen/Qwen3.5-35B-A3B \
	--mem-fraction-static 0.7 \
	--tp 1 \
	--trust-remote-code \
    --cuda-graph-max-bs 128 \
	--host 0.0.0.0 \
	--port 30000 \
	--dtype bfloat16 \
    --reasoning-parser qwen3


2. Regenerate the dataset using the `regenerate_train_data.py` script.
python scripts/regenerate_train_data.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 localhost:30010 localhost:30020 localhost:30030 localhost:30040 localhost:30050 localhost:30060 localhost:30070 \
    --temperature 0.8 \
    --input-file-path /data/jiapingW/pr/SpecForge/cache/dataset/opc_train_first_turn.jsonl \
    --output-file-path ./cache/dataset/opc_train_regen_first_turn.jsonl \
    --resume \
    --is-reasoning-model
"""

import argparse
import json
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using SGLang model servers "
        "(preemption-safe, chunk-based)."
    )

    # model
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument(
        "--is-reasoning-model",
        action="store_true",
        help="Whether the model is a reasoning model",
    )
    model_group.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )
    model_group.add_argument(
        "--thinking-ratio",
        type=float,
        default=None,
        help="Fraction of requests sent with thinking enabled (0 to 1). "
        "Requires --is-reasoning-model. When set, each request randomly "
        "enables or disables thinking based on this ratio. "
        "E.g., 0.7 means 70%% of samples use thinking, 30%% do not.",
    )

    # sampling params
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value sent via extra_body",
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens (default: 4096)",
    )

    # optimization
    opt_group = parser.add_argument_group("optimization")
    opt_group.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Concurrent requests per server (default: 64). "
        "Total concurrency = concurrency * number_of_servers.",
    )

    # data / chunking
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--input-file-path",
        type=str,
        required=True,
        help="Path to input JSONL file",
    )
    data_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for chunk output files (created if needed)",
    )
    data_group.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of input samples per chunk (default: 500). "
        "Determines the granularity of resume: smaller chunks mean less "
        "work is lost on preemption, but more files are created.",
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Max number of samples to process (default: all)",
    )

    # sharding (for multi-pod parallelism)
    shard_group = parser.add_argument_group("sharding")
    shard_group.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This worker's shard index (0-based, default: 0)",
    )
    shard_group.add_argument(
        "--total-shards",
        type=int,
        default=1,
        help="Total number of shards / workers (default: 1)",
    )
    shard_group.add_argument(
        "--chunk-ids",
        type=str,
        default=None,
        help="Comma-separated explicit list of chunk IDs to process "
        "(e.g. '4,5,6,7,12,13'). When set, overrides the default "
        "modulo-based assignment: shard i of --total-shards processes "
        "chunk_ids[j] for j where j %% total_shards == i. Used by the "
        "rescue path to target only the chunks missing from a primary "
        "job. When unset, default behavior is "
        "chunk_id %% total_shards == shard_id.",
    )

    # sglang server
    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        required=True,
        help="Server address(es) as host:port",
    )

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Chunk management
# ---------------------------------------------------------------------------


def chunk_output_path(output_dir: str, chunk_id: int) -> str:
    return os.path.join(output_dir, f"chunk_{chunk_id:06d}.jsonl")


def chunk_error_path(output_dir: str, chunk_id: int) -> str:
    return os.path.join(output_dir, "errors", f"chunk_{chunk_id:06d}_errors.jsonl")


def chunk_done_path(output_dir: str, chunk_id: int) -> str:
    return os.path.join(output_dir, f"chunk_{chunk_id:06d}.done")


def is_chunk_done(output_dir: str, chunk_id: int) -> bool:
    return os.path.exists(chunk_done_path(output_dir, chunk_id))


def write_chunk_done_marker(output_dir: str, chunk_id: int, stats: dict):
    """Write a .done marker with stats. Uses write-then-rename for atomicity."""
    done_path = chunk_done_path(output_dir, chunk_id)
    done_dir = os.path.dirname(done_path)
    try:
        # Write to a temp file then rename for atomicity.
        # On GCSFuse, rename within the same directory is atomic.
        fd, tmp_path = tempfile.mkstemp(dir=done_dir, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(stats, f)
            f.write("\n")
        shutil.move(tmp_path, done_path)
    except OSError:
        # Fallback: direct write (still safe — marker existence is what matters)
        with open(done_path, "w") as f:
            json.dump(stats, f)
            f.write("\n")


def write_chunk_results(
    output_dir: str,
    chunk_id: int,
    results: List[dict],
    errors: List[dict],
    stats: dict,
):
    """
    Write chunk output, error, and done marker files.

    Results and errors are written fully before the .done marker, so if
    the process is killed between writes, the chunk will be retried.
    """
    out_path = chunk_output_path(output_dir, chunk_id)
    err_path = chunk_error_path(output_dir, chunk_id)

    # Write results
    with open(out_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write errors (if any)
    if errors:
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, "w") as f:
            for item in errors:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Mark done last — this is the commit point
    write_chunk_done_marker(output_dir, chunk_id, stats)


# ---------------------------------------------------------------------------
# SGLang interaction (preserved from original)
# ---------------------------------------------------------------------------


def get_random_reasoning_effort() -> str:
    """Get a random reasoning effort level with weighted probabilities."""
    reasoning_efforts = ["low", "medium", "high"]
    weights = [4, 4, 2]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


def compute_context_length(conversations: List[Dict[str, Any]]) -> int:
    """Rough estimate of context length in whitespace-delimited tokens."""
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens

    query_kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.thinking_ratio is not None:
        enable_thinking = random.random() < args.thinking_ratio
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def call_sglang(
    args,
    server_address: str,
    data: Dict[str, Any],
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Regenerate a single sample via an SGLang server."""
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")

    messages = data["conversations"]
    regenerated_messages = []

    # Reject data starting with an assistant message
    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for message in messages:
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            regenerated_messages.append(message)
            query_kwargs = build_query_kwargs(args, regenerated_messages, max_tokens)
            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data
            response_text = resp.choices[0].message.content
            resp_msg = {
                "role": "assistant",
                "content": response_text,
            }
            if args.is_reasoning_model:
                resp_msg["reasoning_content"] = resp.choices[
                    0
                ].message.reasoning_content
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data

    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------


def process_chunk(
    args,
    chunk_id: int,
    chunk_lines: List[str],
    server_addresses: List[str],
) -> Tuple[int, int, dict]:
    """
    Process a single chunk: submit all samples concurrently, collect results,
    write output atomically.

    Returns (success_count, error_count, stats_dict).
    """
    total_concurrency = args.concurrency * len(server_addresses)
    executor = ThreadPoolExecutor(max_workers=total_concurrency)

    # Submit all samples in this chunk
    futures = []
    for i, line in enumerate(chunk_lines):
        data = json.loads(line.strip())
        # Tag with original position within the chunk for ordered output
        data["_chunk_pos"] = i
        server = server_addresses[i % len(server_addresses)]
        future = executor.submit(call_sglang, args, server, data)
        futures.append(future)

    # Collect results in submission order
    results = []
    errors = []
    context_lengths = []

    for future in as_completed(futures):
        regen_data = future.result()
        pos = regen_data.pop("_chunk_pos", -1)

        if regen_data.get("status") == "error":
            regen_data["_chunk_pos"] = pos  # keep for debugging
            errors.append(regen_data)
        else:
            ctx_len = compute_context_length(regen_data.get("conversations", []))
            context_lengths.append(ctx_len)
            regen_data["_chunk_pos"] = pos
            results.append(regen_data)

    executor.shutdown(wait=False)

    # Sort by original position so output is deterministic
    results.sort(key=lambda x: x.pop("_chunk_pos", 0))
    errors.sort(key=lambda x: x.pop("_chunk_pos", 0))

    # Compute stats
    stats = {
        "chunk_id": chunk_id,
        "total": len(chunk_lines),
        "success": len(results),
        "errors": len(errors),
    }
    if context_lengths:
        stats["ctx_len_min"] = min(context_lengths)
        stats["ctx_len_max"] = max(context_lengths)
        stats["ctx_len_avg"] = sum(context_lengths) / len(context_lengths)

    # Write output atomically (results, errors, then .done marker)
    write_chunk_results(args.output_dir, chunk_id, results, errors, stats)

    return len(results), len(errors), stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def plan_chunks(args) -> Tuple[List[Tuple[int, List[str]]], int, int]:
    """
    Read the input file, split into chunks, assign chunks to this shard,
    and filter out chunks already marked done.

    Sharding rules:
      - Default (no --chunk-ids): shard owns chunk_id where
        chunk_id % args.total_shards == args.shard_id.
      - With --chunk-ids: parse the explicit list, then within that list
        shard owns chunk_ids[j] where j % args.total_shards == args.shard_id.
        This is the rescue path: an external orchestrator computes the set
        of missing chunk IDs and divides them among rescue workers.

    Returns (pending_chunks, already_done, total_chunks):
      pending_chunks: [(chunk_id, list_of_input_lines)] this shard must process
      already_done:   number of this shard's chunks that already have .done
                      markers (skipped this run)
      total_chunks:   total chunks across all shards (for logging)
    """
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be > 0, got {args.chunk_size}")
    if args.total_shards <= 0:
        raise ValueError(f"--total-shards must be > 0, got {args.total_shards}")
    if not (0 <= args.shard_id < args.total_shards):
        raise ValueError(
            f"--shard-id must be in [0, {args.total_shards}), got {args.shard_id}"
        )

    # Read input lines once (with optional --num-samples cap)
    with open(args.input_file_path) as f:
        lines = [line for line in f if line.strip()]
    if args.num_samples is not None:
        lines = lines[: args.num_samples]

    # Slice into chunks of CHUNK_SIZE input lines each
    all_chunks: List[Tuple[int, List[str]]] = [
        (cid, lines[cid * args.chunk_size : (cid + 1) * args.chunk_size])
        for cid in range((len(lines) + args.chunk_size - 1) // args.chunk_size)
    ]
    total_chunks = len(all_chunks)

    # Assign chunks to this shard
    if args.chunk_ids is None:
        # Default: stride assignment by chunk_id modulo total_shards
        my_chunks = [
            (cid, payload)
            for cid, payload in all_chunks
            if cid % args.total_shards == args.shard_id
        ]
    else:
        # Rescue path: explicit chunk-id list, then stride within that list
        try:
            requested = [int(x.strip()) for x in args.chunk_ids.split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(
                f"--chunk-ids must be a comma-separated list of integers "
                f"(got {args.chunk_ids!r}): {e}"
            )
        max_cid = total_chunks - 1
        for cid in requested:
            if not (0 <= cid <= max_cid):
                raise ValueError(
                    f"--chunk-ids contains {cid}, which is outside "
                    f"the valid range [0, {max_cid}] for this input file"
                )
        # Sort + dedupe so behavior is deterministic regardless of input order
        requested = sorted(set(requested))
        chunk_by_id = dict(all_chunks)
        my_chunks = [
            (cid, chunk_by_id[cid])
            for j, cid in enumerate(requested)
            if j % args.total_shards == args.shard_id
        ]

    # Drop chunks already marked done
    pending_chunks = [
        (cid, payload)
        for cid, payload in my_chunks
        if not is_chunk_done(args.output_dir, cid)
    ]
    already_done = len(my_chunks) - len(pending_chunks)

    return pending_chunks, already_done, total_chunks


def main():
    args = parse_arguments()

    # Validate sampling params
    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")
    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")
    if args.thinking_ratio is not None:
        if not (0.0 <= args.thinking_ratio <= 1.0):
            raise ValueError("--thinking-ratio must be between 0.0 and 1.0")
        if not args.is_reasoning_model:
            raise ValueError("--thinking-ratio requires --is-reasoning-model")

    print("Configuration:")
    print(f"  Model:             {args.model}")
    print(f"  Max tokens:        {args.max_tokens}")
    print(f"  Concurrency:       {args.concurrency}")
    print(f"  Temperature:       {args.temperature}")
    if args.thinking_ratio is not None:
        print(f"  Thinking ratio:   {args.thinking_ratio:.0%}")
    print(f"  Servers:           {args.server_address}")
    print(f"  Input file:        {args.input_file_path}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Chunk size:        {args.chunk_size}")
    print(f"  Shard:             {args.shard_id}/{args.total_shards}")
    if args.chunk_ids is not None:
        # Truncate the printed list to keep logs readable
        cids_preview = (
            args.chunk_ids
            if len(args.chunk_ids) <= 80
            else (args.chunk_ids[:77] + "...")
        )
        print(f"  Explicit chunks:   {cids_preview}")
    print("-" * 60)

    # Make sure the output dir exists before any chunk write
    os.makedirs(args.output_dir, exist_ok=True)

    # Plan: figure out which chunks this shard owns and which are still pending
    pending_chunks, already_done, total_chunks = plan_chunks(args)

    print(f"Planning:")
    print(f"  Total chunks:      {total_chunks}")
    print(f"  Owned by shard:    {len(pending_chunks) + already_done}")
    print(f"  Already done:      {already_done}")
    print(f"  Pending this run:  {len(pending_chunks)}")
    print("-" * 60)

    if not pending_chunks:
        print(
            f"Shard {args.shard_id}/{args.total_shards}: no pending chunks. "
            f"Nothing to do."
        )
        return

    # Validate server addresses
    valid_servers = []
    for addr in args.server_address:
        dummy = {"conversations": [{"role": "user", "content": "ping"}]}
        result = call_sglang(args, addr, dummy, max_tokens=1)
        if result is not None and result.get("status") != "error":
            valid_servers.append(addr)
        else:
            error_msg = result.get("error", "unknown") if result else "no response"
            print(f"  Warning: server {addr} not available ({error_msg})")

    if not valid_servers:
        raise ValueError("No server address is available")
    print(f"Using {len(valid_servers)} server(s): {valid_servers}")
    print("-" * 60)

    # Process chunks
    total_success = 0
    total_errors = 0
    pbar = tqdm(
        total=len(pending_chunks),
        desc=f"Shard {args.shard_id}",
        initial=0,
    )

    for chunk_id, chunk_lines in pending_chunks:
        pbar.set_postfix(chunk=chunk_id, ok=total_success, err=total_errors)

        success, errors, stats = process_chunk(
            args, chunk_id, chunk_lines, valid_servers
        )
        total_success += success
        total_errors += errors
        pbar.update(1)

        pbar.set_postfix(chunk=chunk_id, ok=total_success, err=total_errors)

    pbar.close()

    # Summary
    print()
    print("=" * 60)
    print("  Processing complete!")
    print("=" * 60)
    print(f"  Shard:            {args.shard_id}/{args.total_shards}")
    print(f"  Chunks processed: {len(pending_chunks)}")
    print(f"  Previously done:  {already_done}")
    print(f"  Successful:       {total_success}")
    print(f"  Errors:           {total_errors}")
    print(f"  Output dir:       {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
