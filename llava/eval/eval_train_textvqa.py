"""
Evaluate training-split TextVQA results against the source JSONL answers.

Unlike ``eval_textvqa.py`` (which uses ``TextVQA_0.5.1_val.json``), this
script builds ground-truth annotations from the training JSONL itself,
whose ``answer`` field comes from the mix665k conversation data.

Usage::

    python eval_train_textvqa.py \
        --question-file llava_textvqa_train_selector.jsonl \
        --result-file answers/train_selector_llava-llama-2-13b-chat-lightning-preview.jsonl \
        --output-file results.csv \
        --separator __sep__
"""

import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator


def _extract_last_answer(answer_text: str, separator: str = "__sep__") -> str:
    """Return the last answer segment when separator-delimited (ICL entries)."""
    if separator in answer_text:
        return answer_text.split(separator)[-1].strip()
    return answer_text.strip()


def _extract_last_qid(qid: str, separator: str = "__sep__") -> str:
    """Return the last question_id segment."""
    if separator in qid:
        return qid.split(separator)[-1].strip()
    return qid.strip()


def eval_train(question_file, result_file, output_file=None, separator="__sep__",
               threshold=0.5, confidence_interval=False):
    """Evaluate training-split results using ground-truth from the question JSONL."""

    # Build annotations from the training JSONL (question_id -> answer)
    annotations = {}
    with open(question_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["question_id"]
            answer = entry.get("answer", "")
            annotations[qid] = answer

    # Load results
    results = []
    with open(result_file, "r") as f:
        for line in f:
            results.append(json.loads(line))

    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)

    evaluator = TextVQAAccuracyEvaluator()
    answer_processor = EvalAIAnswerProcessor()

    pred_list = []
    data_rows = []
    missing = 0

    for result in tqdm(results, desc="Evaluating"):
        raw_qid = result["question_id"]

        # Try full (possibly compound) question_id first, then fall back
        # to the last segment (works for both single and ICL entries).
        gt_answer = annotations.get(raw_qid)
        if gt_answer is None:
            gt_answer = annotations.get(_extract_last_qid(raw_qid, separator))
        if gt_answer is None:
            missing += 1
            continue

        qid = _extract_last_qid(raw_qid, separator)
        gt_answer = _extract_last_answer(gt_answer, separator)
        pred_answer = result["text"]

        # TextVQA evaluator expects gt_answers as a list; wrap single answer
        # in a list of 10 copies (standard TextVQA convention for majority vote)
        gt_answers = [gt_answer] * 10

        processed_pred = answer_processor(pred_answer)
        unique_scores = evaluator._compute_answer_scores(gt_answers)
        score = unique_scores.get(processed_pred, 0.0)

        pred_list.append({
            "pred_answer": pred_answer,
            "gt_answers": gt_answers,
        })

        label = "Correct" if score >= threshold else "Incorrect"
        data_rows.append({
            "id": qid,
            "prompt": result.get("prompt", ""),
            "response": pred_answer,
            "ground-truth": gt_answers,
            "label": label,
        })

    if missing > 0:
        print(f"Warning: {missing} result entries had no matching question in {question_file}")

    # Overall accuracy
    accuracy = evaluator.eval_pred_list(pred_list) if pred_list else 0.0
    print(f"Samples: {len(pred_list)}\nAccuracy: {100. * accuracy:.2f}%\n")

    if confidence_interval and pred_list:
        from llava.eval.utils import bootstrap_confidence_interval
        per_sample_scores = []
        for entry in pred_list:
            processed_pred = answer_processor(entry["pred_answer"])
            unique_scores = evaluator._compute_answer_scores(entry["gt_answers"])
            score = unique_scores.get(processed_pred, 0.0)
            per_sample_scores.append(score)
        lower, upper = bootstrap_confidence_interval(per_sample_scores)
        print(f"Accuracy 95% CI: [{100. * lower:.2f}%, {100. * upper:.2f}%]")

    # Save CSV if requested
    if output_file:
        df = pd.DataFrame(data_rows)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved CSV: {output_file} ({len(data_rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate training-split TextVQA results against source JSONL."
    )
    parser.add_argument(
        "--question-file", type=str, required=True,
        help="Path to the training JSONL (with answer field).",
    )
    parser.add_argument(
        "--result-file", type=str, required=True,
        help="Path to the model's answer JSONL.",
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Output CSV file with Correct/Incorrect labels.",
    )
    parser.add_argument(
        "--separator", type=str, default="__sep__",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Accuracy threshold for Correct/Incorrect label.",
    )
    parser.add_argument(
        "--confidence-interval", action="store_true",
        help="Compute bootstrap confidence interval.",
    )
    args = parser.parse_args()
    eval_train(
        args.question_file, args.result_file, args.output_file,
        args.separator, args.threshold, args.confidence_interval,
    )


if __name__ == "__main__":
    main()
