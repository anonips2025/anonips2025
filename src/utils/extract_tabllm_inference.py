import os
import pickle
import numpy as np
import pandas as pd
import argparse
import json
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helper import get_few_shot_from_csv

def extract_tabllm_inference(dataset, numshot, eval_res_dir="eval_res/tabllm", dataset_dir="dataset"):
    """
    Extracts y_pred and X_synth from t0.p and the synthetic csv, saves as y_pred.npy and X_synth.npy in the same folder.
    """
    result_dir = os.path.join(eval_res_dir, dataset, f"{numshot}_shot")
    t0p_path = os.path.join(result_dir, "t0.p")
    synth_csv_path = os.path.join(dataset_dir, dataset, f"{dataset}_synthetic.csv")
    y_pred_out = os.path.join(result_dir, "y_pred.npy")
    X_synth_out = os.path.join(result_dir, "X_synth.npy")

    with open(t0p_path, "rb") as f:
        t0p = pickle.load(f)
    if "prediction" not in t0p:
        raise KeyError(f"'prediction' not found in {t0p_path}")
    y_pred = np.array(t0p["prediction"])
    np.save(y_pred_out, y_pred)
    print(f"Saved y_pred.npy to {y_pred_out}")

    synth_df = pd.read_csv(synth_csv_path)
    if "label" in synth_df.columns:
        X_synth = synth_df.drop(columns=["label"]).values
    elif "class" in synth_df.columns:
        X_synth = synth_df.drop(columns=["class"]).values
    else:
        X_synth = synth_df.values
    np.save(X_synth_out, X_synth)
    print(f"Saved X_synth.npy to {X_synth_out}")


def extract_tabllm_test_inference(dataset, numshot, eval_res_dir="eval_res/tabllm", seed=0):
    """
    Extracts test inference results and calculates evaluation metrics.
    """
    result_dir = os.path.join(eval_res_dir, dataset, f"{numshot}_shot")
    t0p_test_path = os.path.join(result_dir, "t0_test.p")
    y_test_pred_out = os.path.join(result_dir, "y_test_pred.npy")
    eval_metrics_out = os.path.join(result_dir, "eval_metrics.json")

    # Extract test predictions
    with open(t0p_test_path, "rb") as f:
        t0p_test = pickle.load(f)
    y_test_pred = np.array(t0p_test["prediction"])
    np.save(y_test_pred_out, y_test_pred)
    print(f"Saved y_test_pred.npy to {y_test_pred_out}")

    # Get the true test labels using the same split as TabPFN
    try:
        X_few, y_few, X_train, y_train, X_test, y_test = get_few_shot_from_csv(dataset, numshot, seed)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test.values, y_test_pred)
        
        # Check if predictions are all the same class (which makes ROC-AUC undefined)
        if len(np.unique(y_test_pred)) == 1:
            # All predictions are the same class, set AUC to 50%
            auc = 0.5
        else:
            y_test_prob = np.array(t0p_test["probabilities"])
            # If probabilities are for both classes, take the positive class probability
            if y_test_prob.ndim > 1 and y_test_prob.shape[1] == 2:
                y_test_prob = y_test_prob[:, 1]
            auc = roc_auc_score(y_test.values, y_test_prob)
        
        f1 = f1_score(y_test.values, y_test_pred)
        
        # Calculate TPR and FPR from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test.values, y_test_pred).ravel()
        tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity/Recall)
        fpr = fp / (fp + tn)  # False Positive Rate
        
        # Prepare evaluation metrics
        eval_metrics = {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'f1': float(f1),
            'tpr': float(tpr),
            'fpr': float(fpr)
        }

        # Save evaluation metrics
        with open(eval_metrics_out, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        
        print(f"Saved eval_metrics.json to {eval_metrics_out}")
        print(f"Test set performance: AUC={auc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
        
    except Exception as e:
        print(f"Error calculating test metrics for {dataset}: {str(e)}")


def extract_tabllm_full(dataset, numshot, eval_res_dir="eval_res/tabllm", dataset_dir="dataset", seed=0):
    """
    Extract both synthetic and test inference results and calculate metrics.
    """
    # Extract synthetic data inference
    extract_tabllm_inference(dataset, numshot, eval_res_dir, dataset_dir)
    
    # Extract test data inference and calculate metrics
    extract_tabllm_test_inference(dataset, numshot, eval_res_dir, seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--numshot", type=int, required=True)
    parser.add_argument("--eval_res_dir", type=str, default="eval_res/tabllm")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["synthetic", "test", "full"], default="full",
                        help="What to extract: 'synthetic' for synthetic data only, 'test' for test data only, 'full' for both")
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        extract_tabllm_inference(args.dataset, args.numshot, args.eval_res_dir, args.dataset_dir)
    elif args.mode == "test":
        extract_tabllm_test_inference(args.dataset, args.numshot, args.eval_res_dir, args.seed)
    else:  # full
        extract_tabllm_full(args.dataset, args.numshot, args.eval_res_dir, args.dataset_dir, args.seed)

if __name__ == "__main__":
    main()
