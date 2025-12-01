import json
import pandas as pd
import argparse
import os


def load_json(file_path, sep="__sep__"):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Handle predictions format
    if isinstance(data, list):
        data = {item["questionId"].split(sep)[-1]: item["prediction"] for item in data}
    return data


def main():
    parser = argparse.ArgumentParser(description="Create DataFrame with QA results.")
    parser.add_argument("--tier", default="val", type=str, help="Tier, e.g. train, val")
    parser.add_argument(
        "--questions",
        default="{tier}_questions.json",
        type=str,
        help="Questions file name format.",
    )
    parser.add_argument(
        "--predictions",
        default="{tier}_predictions.json",
        type=str,
        help="Answers file name format.",
    )
    parser.add_argument(
        "--output", type=str, default="results.csv", help="Output CSV file name."
    )
    parser.add_argument(
        "--sep", default="__sep__", type=str, help="Separator used in IDs (if any)."
    )
    args = parser.parse_args()

    # Load questions and predictions
    print("Loading questions...")
    if not os.path.isfile(args.questions):
        raise FileNotFoundError(f"Questions file not found: {args.questions}")
    with open(args.questions, "r") as f:
        questions = json.load(f)

    print("Loading predictions...")
    if not os.path.isfile(args.predictions):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")
    predictions = load_json(args.predictions, sep=args.sep)

    # Prepare data for DataFrame
    data = []
    missing_predictions = []
    for qid, qdata in questions.items():
        prompt = qdata["question"]
        ground_truth = qdata["answer"]
        response = predictions.get(qid)
        if response is None:
            missing_predictions.append(qid)
            response = "N/A"
            label = "No Prediction"
        else:
            label = "Correct" if response == ground_truth else "Incorrect"
        data.append(
            {
                "id": qid,
                "prompt": prompt,
                "response": response,
                "ground-truth": ground_truth,
                "label": label,
            }
        )

    if missing_predictions:
        print(f"Warning: Missing predictions for {len(missing_predictions)} questions.")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
