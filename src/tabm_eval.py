import argparse
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helper import get_few_shot_from_csv
from src.ttnet.ttnet_wrapper import TTNetPreprocessor
from baselines.utils_baselines import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--numshot", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    # Output directory
    save_dir = f"eval_res/tabm/{args.dataset}/{args.numshot}_shot/"
    results_path = os.path.join(save_dir, "eval_metrics.json")
    if os.path.exists(results_path):
        print(f"Output already exists at {save_dir}. Skipping run.")
        return

    # Load Data
    X_few, y_few, X_train, y_train, X_test, y_test = get_few_shot_from_csv(args.dataset, args.numshot, args.seed)
    
    synth_path = f"dataset/{args.dataset}/{args.dataset}_synthetic.csv"
    synth_df = pd.read_csv(synth_path)
    X_synth = synth_df.drop(columns=["class"])
    
    # Preprocess data
    info_path = f"dataset/{args.dataset}/{args.dataset}.info"
    preprocessor = TTNetPreprocessor(info_path=info_path)
    
    X_original_full = pd.concat([X_train, X_test], ignore_index=True)
    preprocessor.fit(X_original_full)

    X_few_proc, _ = preprocessor.transform(X_few)
    X_synth_proc, _ = preprocessor.transform(X_synth)

    # Convert to tensors
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    X_few_tensor = torch.as_tensor(X_few_proc, device=device, dtype=torch.float32)
    y_few_tensor = torch.as_tensor(y_few.values, device=device, dtype=torch.long)
    X_synth_tensor = torch.as_tensor(X_synth_proc, device=device, dtype=torch.float32)
    
    # Process test data for evaluation
    X_test_proc, _ = preprocessor.transform(X_test)
    X_test_tensor = torch.as_tensor(X_test_proc, device=device, dtype=torch.float32)

    # Model setup
    n_num_features = X_few_proc.shape[1]
    n_classes = len(np.unique(y_few))

    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=[],
        d_out=n_classes,
        num_embeddings=None,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training
    @torch.inference_mode()
    def apply_model(x_num):
        return model(x_num, None).squeeze(-1).float()

    def loss_fn(y_pred, y_true):
        y_pred = y_pred.flatten(0, 1)
        y_true = y_true.repeat_interleave(model.backbone.k)
        return nn.functional.cross_entropy(y_pred, y_true)

    for epoch in range(500): # Max epochs
        model.train()
        optimizer.zero_grad()
        
        y_pred_train = model(X_few_tensor, None).squeeze(-1).float()
        loss = loss_fn(y_pred_train, y_few_tensor)
        loss.backward()
        optimizer.step()

    # Inference
    model.eval()
    with torch.inference_mode():
        # Inference on synthetic data
        y_pred_synth = apply_model(X_synth_tensor)
        y_pred_synth = torch.softmax(y_pred_synth, dim=-1).mean(1)
        y_pred = y_pred_synth.argmax(1).cpu().numpy()
        
        # Inference on test data
        y_pred_test_logits = apply_model(X_test_tensor)
        y_pred_test_probs = torch.softmax(y_pred_test_logits, dim=-1).mean(1)
        y_test_pred = y_pred_test_probs.argmax(1).cpu().numpy()
        y_test_probs = y_pred_test_probs.cpu().numpy()
    
    # Evaluate performance on test set
    accuracy = accuracy_score(y_test.values, y_test_pred)
    
    # Check if predictions are all the same class (which makes ROC-AUC undefined)
    if len(np.unique(y_test_pred)) == 1:
        # All predictions are the same class, set AUC to 50%
        auc = 0.5
        print(f"Warning: All test predictions are class {y_test_pred[0]}, setting AUC to 0.5")
    else:
        auc = roc_auc_score(y_test.values, y_test_probs[:, 1])
    
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

    # Save results
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
