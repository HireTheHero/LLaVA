import argparse
import json
import os
import re

import pandas as pd
from tqdm import tqdm

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator
from llava.eval.eval_textvqa import qid_prompt_processor


def create_dataframe(
    annotation_file, result_file, output_file, separator="__sep__", threshold=0.5
):
    # Load annotations
    print("Loading annotations...")
    with open(annotation_file, "r") as f:
        annotations_json = json.load(f)["data"]
    print(f"Loaded {len(annotations_json)} annotations.")
    annotations = {
        (annotation["image_id"], annotation["question"].lower()): annotation
        for annotation in annotations_json
    }
    print(f"Processed {len(annotations)} annotations with unique keys.")
    # Load results
    print("Loading results...")
    with open(result_file, "r") as f:
        results = [json.loads(line) for line in f]
    print(f"Loaded {len(results)} results.")
    data = []
    evaluator = TextVQAAccuracyEvaluator()
    answer_processor = EvalAIAnswerProcessor()
    missing_annotations = []
    for result in tqdm(results, desc="Processing results"):
        try:
            key = qid_prompt_processor(
                result["question_id"], result["prompt"], separator=separator
            )
            annotation = annotations.get(key)
            if annotation is None:
                missing_annotations.append(key)
                continue
            pred_answer = result["text"]
            gt_answers = annotation["answers"]
            # Compute accuracy score
            processed_pred_answer = answer_processor(pred_answer)
            unique_answer_scores = evaluator._compute_answer_scores(gt_answers)
            score = unique_answer_scores.get(processed_pred_answer, 0.0)
            # Determine label based on threshold
            label = "Correct" if score >= threshold else "Incorrect"
            data.append(
                {
                    "id": key[0],
                    "prompt": result["prompt"],
                    "response": pred_answer,
                    "ground-truth": gt_answers,
                    "label": label,
                }
            )
        except Exception as e:
            raise e
            # print(f"Error processing result with question_id {result['question_id']}: {e}")
            # continue
    if missing_annotations:
        print(f"Warning: Missing annotations for {len(missing_annotations)} questions.")
    # Create DataFrame
    df = pd.DataFrame(data)
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"DataFrame saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create DataFrame from QA results.")
    parser.add_argument(
        "--annotation-file",
        type=str,
        required=True,
        help="Path to annotation JSON file.",
    )
    parser.add_argument(
        "--result-file", type=str, required=True, help="Path to result JSONL file."
    )
    parser.add_argument(
        "--output-file", type=str, default="results.csv", help="Output CSV file name."
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="__sep__",
        help="Separator used in IDs (if any).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Accuracy threshold for label assignment.",
    )
    args = parser.parse_args()
    create_dataframe(
        args.annotation_file,
        args.result_file,
        args.output_file,
        args.separator,
        args.threshold,
    )


if __name__ == "__main__":
    main()
