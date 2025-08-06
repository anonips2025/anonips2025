import argparse
import os
import json
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*", category=FutureWarning)

from carte_ai import CARTEClassifier
from huggingface_hub import hf_hub_download
from carte_ai import Table2GraphTransformer

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helper import get_few_shot_from_csv
from baselines.utils_baselines import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--numshot", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dataset_name = args.dataset
    num_shot = args.numshot
    seed = args.seed
    device = args.device

    set_seed(seed)

    save_dir = f"eval_res/carte/{dataset_name}/{num_shot}_shot/"
    results_path = os.path.join(save_dir, "eval_metrics.json")
    if os.path.exists(results_path):
        print(f"Output already exists at {save_dir}. Skipping run.")
        return

    X_few, y_few, X_train, y_train, X_test, y_test = get_few_shot_from_csv(dataset_name, num_shot, seed)
    X_few = X_few.reset_index(drop=True)
    y_few = y_few.reset_index(drop=True)
    print(f"Sampled {len(y_few)} few-shot examples: {np.bincount(y_few.to_numpy())}")

    synth_path = f"dataset/{dataset_name}/{dataset_name}_synthetic.csv"
    synth_df = pd.read_csv(synth_path)
    X_synth = synth_df.drop(columns=["class"])
    y_synth = synth_df["class"].values

    for col in X_few.columns:
        if col in X_synth.columns:
            if X_synth[col].dtype in ['int64', 'float64']:
                X_few[col] = pd.to_numeric(X_few[col], errors='coerce')
    
    model_path = hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin")
    preprocessor = Table2GraphTransformer(fasttext_model_path=model_path)

    X_few_graph = preprocessor.fit_transform(X_few, y=y_few.reset_index(drop=True).to_numpy())
    X_synth_graph = preprocessor.transform(X_synth)
    X_test_graph = preprocessor.transform(X_test)

    clf = CARTEClassifier(device=device)
    
    # For very small datasets, CARTE might fail with internal validation split
    # Try to fit, and if it fails due to insufficient samples, skip this configuration
    try:
        clf.fit(X_few_graph, y_few)
    except ValueError as e:
        if "test_size" in str(e) and "greater or equal to the number of classes" in str(e):
            print(f"Skipping {dataset_name} with {num_shot} shots due to insufficient samples for CARTE validation split")
            return
        else:
            raise e

    # Inference on synthetic data
    y_pred = clf.predict(X_synth_graph)
    
    # Inference on test data
    y_test_pred = clf.predict(X_test_graph)
    y_test_probs = clf.predict_proba(X_test_graph)
    
    # Evaluate performance on test set
    accuracy = accuracy_score(y_test.values, y_test_pred)
    
    # Check if predictions are all the same class (which makes ROC-AUC undefined)
    if len(np.unique(y_test_pred)) == 1:
        # All predictions are the same class, set AUC to 50%
        auc = 0.5
        print(f"Warning: All test predictions are class {y_test_pred[0]}, setting AUC to 0.5")
    else:
        # Handle probability array dimensions - CARTE might return 1D or 2D arrays
        if y_test_probs.ndim == 2 and y_test_probs.shape[1] == 2:
            # Binary classification with 2D probabilities - take positive class
            auc = roc_auc_score(y_test.values, y_test_probs[:, 1])
        elif y_test_probs.ndim == 1:
            # 1D probabilities - use as is
            auc = roc_auc_score(y_test.values, y_test_probs)
        else:
            # Fallback - use predictions for AUC (less accurate)
            auc = roc_auc_score(y_test.values, y_test_pred)
    
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

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(save_dir, "X_synth.npy"), X_synth)
    np.save(os.path.join(save_dir, "y_test_pred.npy"), y_test_pred)
    
    # Save evaluation metrics
    with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)
    
    with open(os.path.join(save_dir, "commandline_args.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Saved inference results to {save_dir}")
    print(f"Test set performance: AUC={auc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")

if __name__ == "__main__":
    main()
