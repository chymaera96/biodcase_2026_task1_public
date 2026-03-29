"""
Code to evaluate model predictions
Usage:
python evaluate.py --predictions-fp=/path/to/predictions.csv --ground-truth-fp=/path/to/ground/truth.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-fp", type=str, required=True, help="path to predictions csv")
    parser.add_argument("--ground-truth-fp", type=str, required=True, help="path to ground truth csv")
    args = parser.parse_args()
    return args


def evaluate_from_files(predictions_fp, ground_truth_fp):
    """
    Evaluate predictions against ground truth annotations and save the results.

    Args:
    predictions_fp (str): File path to the CSV file containing predictions.
    ground_truth_fp (str): File path to the CSV file containing ground truth annotations.

    Returns:
    dict: A dictionary containing evaluation results.
    """
    predictions = pd.read_csv(predictions_fp)
    annotations = pd.read_csv(ground_truth_fp)

    results = evaluate(predictions, annotations)
    print(f"Average MSE (sec^2): {results['overall_mse']}")
    print(f"Average MAE (ms): {results['overall_mae_ms']}")

    output_fp = predictions_fp.replace(".csv", "_evaluation.yaml")
    with open(output_fp, "w") as f:
        yaml.dump(results, f)

    print(f"Writing results to {output_fp}")
    return results


def evaluate(predictions, annotations):
    """
    Compute the mean squared error (MSE) between predicted and annotated keypoints for each file.

    Args:
    predictions (pd.DataFrame): A DataFrame containing predicted keypoints with columns "Filename",
                                "Time Channel 0", and "Time Channel 1".
    annotations (pd.DataFrame): A DataFrame containing ground truth keypoints with the same columns.

        Returns:
        dict: A dictionary mapping filenames to per-file metrics, with additional overall averages.
    """

    filenames = sorted(annotations["Filename"].unique())
    # Per-file entries remain detailed dicts for newer use.
    # Add summary/alias keys at the end for backward compatibility.
    results = {}

    for filename in filenames:
        predictions_sub = predictions[predictions["Filename"] == filename]
        annotations_sub = annotations[annotations["Filename"] == filename]

        keypoints = sorted(annotations_sub["Time Channel 0"].unique())
        squared_errors = []
        abs_errors = []
        for keypoint in keypoints:
            pred_row = predictions_sub[predictions_sub["Time Channel 0"] == keypoint]
            anno_row = annotations_sub[annotations_sub["Time Channel 0"] == keypoint]

            if len(pred_row) > 1:
                # Make sure there is at most one prediction for each keypoint
                warnings.warn(f"Multiple predictions for one keypoint in file {filename}, using the first prediction")
                predtime = pred_row["Time Channel 1"].iloc[0]
            elif len(pred_row) == 0:
                # Make sure there is at least one prediction for each keypoint
                warnings.warn(f"Missing predictions for some keypoints in file {filename}, using non-aligned value")
                predtime = keypoint
            else:
                predtime = pred_row["Time Channel 1"].iloc[0]

            annotime = anno_row["Time Channel 1"].iloc[0]

            squared_error = (predtime - annotime) ** 2
            squared_errors.append(squared_error)

            abs_error = abs(predtime - annotime)
            abs_errors.append(abs_error)

        mse = float(np.mean(squared_errors))
        mae_ms = float(np.mean(abs_errors) * 1000.0)

        results[filename] = {
            "mse": mse,
            "mae_ms": mae_ms,
        }

    all_mses = [results[x]["mse"] for x in filenames]
    all_maes = [results[x]["mae_ms"] for x in filenames]

    results["overall_mse"] = float(np.mean(all_mses)) if len(all_mses) else 0.0
    results["overall_mae_ms"] = float(np.mean(all_maes)) if len(all_maes) else 0.0

    # Backward-compatible aliases (older code expected MSE only).
    results["overall"] = results["overall_mse"]
    results["files"] = {fn: results[fn]["mse"] for fn in filenames}
    results["files_mae_ms"] = {fn: results[fn]["mae_ms"] for fn in filenames}
    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate_from_files(args.predictions_fp, args.ground_truth_fp)
