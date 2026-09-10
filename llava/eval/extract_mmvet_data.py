import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def create_dataframe(
    mmvet_metadata_file,
    result_file,
    grade_file,
    output_file,
    threshold=0.5,
):
    """Create a per-question CSV from MM-Vet evaluation results.

    Reads MM-Vet metadata (questions + ground truth), model predictions,
    and GPT grade results to produce a CSV matching the TextVQA/GQA format
    used by the downstream probe analysis pipeline.
    """
    # Load MM-Vet metadata (questions, answers, capabilities)
    print("Loading MM-Vet metadata...")
    with open(mmvet_metadata_file, "r") as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} questions from metadata.")

    # Load model predictions
    print("Loading model predictions...")
    with open(result_file, "r") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions.")

    # Load GPT grade results
    print("Loading GPT grade results...")
    with open(grade_file, "r") as f:
        grade_results = json.load(f)
    print(f"Loaded {len(grade_results)} grade results.")

    data = []
    missing_predictions = []
    missing_grades = []

    for qid, qdata in tqdm(metadata.items(), desc="Processing results"):
        question = qdata["question"]
        answer = qdata["answer"]

        # Get model prediction
        prediction = predictions.get(qid)
        if prediction is None:
            missing_predictions.append(qid)
            prediction = "N/A"

        # Get GPT grade
        grade = grade_results.get(qid)
        if grade is None:
            missing_grades.append(qid)
            score = 0.0
        else:
            # Average across multiple runs
            scores = grade.get("score", [0.0])
            score = float(np.mean(scores))

        # Determine label based on threshold
        label = "Correct" if score >= threshold else "Incorrect"

        data.append(
            {
                "id": qid,
                "prompt": question,
                "response": prediction,
                "ground-truth": answer,
                "label": label,
            }
        )

    if missing_predictions:
        print(
            f"Warning: Missing predictions for {len(missing_predictions)} questions."
        )
    if missing_grades:
        print(f"Warning: Missing grades for {len(missing_grades)} questions.")

    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"DataFrame saved to {output_file} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Create per-question CSV from MM-Vet evaluation results."
    )
    parser.add_argument(
        "--mmvet-metadata",
        type=str,
        required=True,
        help="Path to mm-vet.json metadata file.",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="Path to model predictions JSON file.",
    )
    parser.add_argument(
        "--grade-file",
        type=str,
        required=True,
        help="Path to GPT grade results JSON file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="mmvet_results.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for Correct/Incorrect label assignment.",
    )
    args = parser.parse_args()
    create_dataframe(
        args.mmvet_metadata,
        args.result_file,
        args.grade_file,
        args.output_file,
        args.threshold,
    )


if __name__ == "__main__":
    main()
