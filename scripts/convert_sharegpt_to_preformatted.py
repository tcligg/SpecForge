"""Convert ShareGPT-format JSONL to preformatted text JSONL.

This is needed for models whose tokenizer chat template strips content
(e.g. Gemma4's strip_thinking macro removes <|channel>...<channel|> blocks).
By converting to preformatted text, the training pipeline skips
apply_chat_template entirely, preserving all tokens as-is.

Inputs can be individual JSONL files or directories of JSONL chunks
(e.g. from regeneration pipelines). Directories are concatenated in
sorted order before conversion.

Usage:
    # Single file:
    python scripts/convert_sharegpt_to_preformatted.py \
        --input outputs/dataset/ultrachat_regen_gemma4.jsonl \
        --output outputs/dataset/ultrachat_regen_gemma4_preformatted.jsonl \
        --format gemma4

    # Multiple files and directories:
    python scripts/convert_sharegpt_to_preformatted.py \
        --input outputs/dataset/ultrachat_regen_gemma4.jsonl \
               ~/gcs/pyc-eagle-data/datasets/gemma4-26b-regen/perfectblend_train \
               ~/gcs/pyc-eagle-data/datasets/gemma4-26b-regen/translate-bp-dataset_train \
        --output outputs/dataset/combined_preformatted.jsonl \
        --format gemma4
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

from tqdm import tqdm


def format_gemma4(
    conversations: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    enable_thinking: bool = True,
) -> str:
    """Render a ShareGPT conversation to Gemma4 format, preserving all tokens.

    Produces the same output as the Gemma4 tokenizer chat template but
    without stripping <|channel>...<channel|> blocks from model content.

    Args:
        conversations: List of {"role": ..., "content": ...} dicts.
        system_prompt: Optional system prompt text. If the first message has
            role "system", it takes precedence over this argument.
        enable_thinking: If True, inject <|think|> in the system turn.

    Returns:
        The fully formatted conversation string.
    """
    parts = ["<bos>"]
    remaining = conversations

    # Determine system content
    sys_content = ""
    if remaining and remaining[0]["role"] in ("system", "developer"):
        sys_content = remaining[0]["content"].strip()
        remaining = remaining[1:]
    elif system_prompt:
        sys_content = system_prompt.strip()

    # Emit system turn if needed (enable_thinking or system content present)
    if enable_thinking or sys_content:
        parts.append("<|turn>system\n")
        if enable_thinking:
            parts.append("<|think|>\n")
        if sys_content:
            parts.append(sys_content)
        parts.append("<turn|>\n")

    # Emit user/assistant turns
    for msg in remaining:
        role = msg["role"]
        content = msg.get("content") or ""

        if role == "assistant":
            role_tag = "model"
        else:
            role_tag = role

        parts.append(f"<|turn>{role_tag}\n")
        if role_tag == "model":
            # Preserve content exactly (including <|channel> blocks)
            parts.append(content)
        else:
            parts.append(content.strip())
        parts.append("<turn|>\n")

    return "".join(parts)


def format_gemma3(
    conversations: List[Dict[str, str]],
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Render a ShareGPT conversation to Gemma3 format.

    Gemma3 folds the system prompt into the first user turn.

    Args:
        conversations: List of {"role": ..., "content": ...} dicts.
        system_prompt: Default system prompt. Overridden if the first
            message has role "system".

    Returns:
        The fully formatted conversation string.
    """
    parts = ["<bos>"]
    remaining = conversations

    # Handle system message
    if remaining and remaining[0]["role"] == "system":
        system_prompt = remaining[0]["content"].strip()
        remaining = remaining[1:]

    first_user = True
    for msg in remaining:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            role_tag = "model"
        else:
            role_tag = role

        parts.append(f"<start_of_turn>{role_tag}\n")
        if role_tag == "user" and first_user and system_prompt:
            parts.append(f"{system_prompt}\n\n{content.strip()}")
            first_user = False
        elif role_tag == "model":
            parts.append(content)
        else:
            parts.append(content.strip())
        parts.append("<end_of_turn>\n")

    return "".join(parts)


FORMATTERS = {
    "gemma4": format_gemma4,
    "gemma3": format_gemma3,
}


def resolve_input_files(paths: List[str]) -> List[str]:
    """Resolve a list of paths to individual JSONL files.

    Each path can be:
      - A single .jsonl file
      - A directory containing .jsonl chunk files (sorted by name)

    Returns a flat list of resolved file paths.
    """
    resolved = []
    for path in paths:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            chunks = sorted(glob.glob(os.path.join(path, "*.jsonl")))
            if not chunks:
                print(
                    f"WARNING: No .jsonl files found in directory: {path}",
                    file=sys.stderr,
                )
            else:
                print(f"  Resolved directory {path} -> {len(chunks)} chunk(s)")
            resolved.extend(chunks)
        elif os.path.isfile(path):
            resolved.append(path)
        else:
            print(f"ERROR: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT JSONL to preformatted text JSONL"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input ShareGPT JSONL file(s) or directory(ies) of JSONL chunks",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output preformatted JSONL file"
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=list(FORMATTERS.keys()),
        help="Target model format",
    )
    args = parser.parse_args()

    formatter = FORMATTERS[args.format]

    input_files = resolve_input_files(args.input)
    if not input_files:
        print("ERROR: No input files resolved.", file=sys.stderr)
        sys.exit(1)

    num_output = 0
    num_skipped = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w") as fout:
        for input_file in tqdm(
            input_files, desc="Files", disable=len(input_files) == 1
        ):
            with open(input_file, "r") as fin:
                for line in fin:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        num_skipped += 1
                        continue
                    conversations = record.get("conversations", [])

                    if not conversations:
                        num_skipped += 1
                        continue

                    text = formatter(conversations)

                    out_record = {"text": text}
                    if "id" in record:
                        out_record["id"] = record["id"]

                    fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    num_output += 1

    print(f"Done. Output: {num_output}, Skipped: {num_skipped}")


if __name__ == "__main__":
    main()
