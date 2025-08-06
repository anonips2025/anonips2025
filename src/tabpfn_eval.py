import argparse
import os
import json
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

import sys
from pathlib import Path

# Add the project root to the Python path
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

    # Output directory
    save_dir = f"eval_res/tabpfn/{dataset_name}/{num_shot}_shot/"
    args_path = os.path.join(save_dir, "commandline_args.txt")
    if os.path.exists(args_path):
        print(f"Output already exists at {save_dir}. Skipping run.")
        return

    # Few-shot sample from train using helper
    X_few, y_few, X_train, y_train, X_test, y_test  = get_few_shot_from_csv(dataset_name, num_shot, seed)
    print(f"Sampled {len(y_few)} few-shot examples: {np.bincount(y_few.to_numpy())}")

    # Load synthetic test set
    synth_path = f"dataset/{dataset_name}/{dataset_name}_synthetic.csv"
    synth_df = pd.read_csv(synth_path)
    X_synth = synth_df.drop(columns=["class"]).values
    y_synth = synth_df["class"].values

    # Train TabPFN
    clf = TabPFNClassifier(device=device)
    clf.fit(X_few.values, y_few.values)

    # Inference on synthetic data
    y_pred = clf.predict(X_synth)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(save_dir, "X_synth.npy"), X_synth)
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved inference results to {save_dir}")

if __name__ == "__main__":
    main()