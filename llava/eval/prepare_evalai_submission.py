"""Prepare EvalAI submission files for VQAv2 and VizWiz.

Unified script with a flat-file interface compatible with
analyze_dataset.sh answer file paths (no merge.jsonl assumption).
"""

import argparse
import json
import os

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert model predictions to EvalAI submission format."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["vqav2", "vizwiz"],
        help="Target dataset (determines output format).",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="Path to model predictions JSONL file.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to question/annotation JSONL file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to write submission JSON.",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="__sep__",
        help="Separator used in composite question IDs.",
    )
    return parser.parse_args()


def load_results(result_file):
    """Load JSONL results, tolerating malformed lines."""
    results = []
    error_lines = 0
    for line in open(result_file):
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            error_lines += 1
    return results, error_lines


def prepare_vqav2(results, questions, separator):
    """Prepare VQAv2 submission: [{"question_id": int, "answer": str}, ...]"""
    result_map = {
        int(str(r["question_id"]).split(separator)[-1]): r["text"] for r in results
    }
    answer_processor = EvalAIAnswerProcessor()
    all_answers = []
    for q in questions:
        qid = q["question_id"]
        if qid not in result_map:
            all_answers.append({"question_id": qid, "answer": ""})
        else:
            all_answers.append(
                {"question_id": qid, "answer": answer_processor(result_map[qid])}
            )
    return all_answers


def prepare_vizwiz(results, questions, separator):
    """Prepare VizWiz submission: [{"image": str, "answer": str}, ...]"""
    result_map = {
        str(r["question_id"]).split(separator)[-1]: r["text"] for r in results
    }
    answer_processor = EvalAIAnswerProcessor()
    all_answers = []
    missing = 0
    for q in questions:
        qid = q["question_id"]
        if qid not in result_map:
            all_answers.append({"image": q["image"], "answer": ""})
            missing += 1
        else:
            all_answers.append(
                {"image": q["image"], "answer": answer_processor(result_map[qid])}
            )
    if missing:
        print(f"Warning: {missing} questions had no matching prediction.")
    return all_answers


if __name__ == "__main__":
    args = parse_args()

    # Load predictions and questions
    results, error_lines = load_results(args.result_file)
    questions = [json.loads(line) for line in open(args.question_file)]
    print(
        f"total results: {len(results)}, total questions: {len(questions)}, "
        f"error_lines: {error_lines}"
    )

    # Format for target dataset
    if args.dataset == "vqav2":
        all_answers = prepare_vqav2(results, questions, args.separator)
    elif args.dataset == "vizwiz":
        all_answers = prepare_vizwiz(results, questions, args.separator)

    # Write submission file
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_answers, f)
    print(f"Submission file written to {args.output_file} ({len(all_answers)} entries)")
