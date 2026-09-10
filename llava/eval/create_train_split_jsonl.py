"""
Create training-split JSONL for the ZSL/ICL selector probe.

Extracts task-specific entries from ``llava_v1_5_mix665k.json``,
reshapes them into the eval JSONL format expected by ``model_vqa_loader.py``,
and partitions them into a *query subset* (used as probe training data) and a
*reference subset* (reserved for ICL example selection by
``multiple_input_dataset.py``).

Usage::

    python create_train_split_jsonl.py \
        --train-data /path/to/llava_v1_5_mix665k.json \
        --task textvqa \
        --eval-jsonl /path/to/llava_textvqa_val_v051_ocr.jsonl \
        --output-dir /path/to/eval/textvqa \
        --amplify-ratio 2 \
        --seed 42
"""

import argparse
import json
import os
import random
import re
import sys


DEFAULT_IMAGE_TOKEN = "<image>"


def load_mix665k_task_subset(train_data_path: str, task: str):
    """Load mix665k and filter to the task-specific subset."""
    with open(train_data_path, "r") as f:
        data = json.load(f)

    task_lower = task.lower()
    subset = []
    for idx, entry in enumerate(data):
        image = entry.get("image")
        if image is None:
            continue
        source = image.split("/")[0]
        if source == task_lower:
            entry["_mix665k_idx"] = idx
            subset.append(entry)
    return subset


def convert_to_eval_jsonl(entry, query_id: str):
    """Reshape a mix665k entry into the eval JSONL format.

    Returns a dict with keys: question_id, text, image, answer.
    """
    conversations = entry["conversations"]
    text = conversations[0]["value"]
    # Strip the <image> token -- add_image_token() re-adds it at inference time.
    # Without this, the prompt would contain duplicate image tokens.
    text = re.sub(DEFAULT_IMAGE_TOKEN, "", text).strip()
    answer = conversations[1]["value"] if len(conversations) > 1 else ""
    # Store only the basename -- the directory prefix (e.g. "textvqa/train_images/")
    # is already provided by image_folder in llava_eval.yaml
    image = os.path.basename(entry["image"])

    return {
        "question_id": query_id,
        "text": text,
        "image": image,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create training-split JSONL for the ZSL/ICL selector probe."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to llava_v1_5_mix665k.json",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g. textvqa, gqa)",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        required=True,
        help="Path to the existing eval JSONL (to determine N_eval for ratio).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the generated files.",
    )
    parser.add_argument(
        "--amplify-ratio",
        type=float,
        default=2.0,
        help="Target training size = N_eval * amplify_ratio (capped at 80%% of pool).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-ref-fraction",
        type=float,
        default=0.2,
        help="Minimum fraction of the pool reserved for ICL references.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Load task-specific subset from mix665k ---
    print(f"Loading mix665k from {args.train_data}...")
    subset = load_mix665k_task_subset(args.train_data, args.task)
    pool_size = len(subset)
    print(f"Found {pool_size} entries for task '{args.task}'")

    if pool_size == 0:
        print(f"Error: No entries found for task '{args.task}' in mix665k.", file=sys.stderr)
        sys.exit(1)

    # --- Determine target query size ---
    n_eval = sum(1 for _ in open(args.eval_jsonl, "r"))
    target_query_size = int(n_eval * args.amplify_ratio)
    max_query_size = int(pool_size * (1.0 - args.max_ref_fraction))
    query_size = min(target_query_size, max_query_size)

    print(f"Eval JSONL has {n_eval} samples")
    print(f"Target query size (N_eval * {args.amplify_ratio}): {target_query_size}")
    print(f"Max query size (pool * {1.0 - args.max_ref_fraction:.0%}): {max_query_size}")
    print(f"Actual query size: {query_size}")

    # --- Partition: shuffle and split ---
    random.shuffle(subset)
    query_entries = subset[:query_size]
    ref_entries = subset[query_size:]

    print(f"Query subset: {len(query_entries)} samples")
    print(f"Reference subset: {len(ref_entries)} samples (for ICL matching)")

    # --- Convert query subset to eval JSONL ---
    os.makedirs(args.output_dir, exist_ok=True)

    task_lower = args.task.lower()
    jsonl_path = os.path.join(args.output_dir, f"llava_{task_lower}_train_selector.jsonl")

    with open(jsonl_path, "w") as f:
        for i, entry in enumerate(query_entries):
            query_id = f"train_{entry['_mix665k_idx']}"
            record = convert_to_eval_jsonl(entry, query_id)
            f.write(json.dumps(record) + "\n")

    print(f"Wrote query JSONL: {jsonl_path} ({len(query_entries)} lines)")

    # --- Write reference IDs (mix665k indices to keep for ICL) ---
    ref_ids_path = os.path.join(args.output_dir, f"llava_{task_lower}_train_selector_ref_ids.txt")
    with open(ref_ids_path, "w") as f:
        for entry in ref_entries:
            f.write(f"{entry['_mix665k_idx']}\n")

    print(f"Wrote reference IDs: {ref_ids_path} ({len(ref_entries)} lines)")

    # --- Also write query IDs (for --exclude-ids in multiple_input_dataset.py) ---
    query_ids_path = os.path.join(args.output_dir, f"llava_{task_lower}_train_selector_query_ids.txt")
    with open(query_ids_path, "w") as f:
        for entry in query_entries:
            f.write(f"{entry['_mix665k_idx']}\n")

    print(f"Wrote query IDs (for exclusion): {query_ids_path} ({len(query_entries)} lines)")

    # --- Summary ---
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Task:              {args.task}")
    print(f"Pool size:         {pool_size}")
    print(f"Query subset:      {len(query_entries)} (probe training)")
    print(f"Reference subset:  {len(ref_entries)} (ICL matching)")
    print(f"Query JSONL:       {jsonl_path}")
    print(f"Reference IDs:     {ref_ids_path}")
    print(f"Query IDs:         {query_ids_path}")


if __name__ == "__main__":
    main()
