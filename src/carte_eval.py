import argparse
import os
import json
import numpy as np
import pandas as pd
import warnings

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
    args_path = os.path.join(save_dir, "commandline_args.txt")
    if os.path.exists(args_path):
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

    clf = CARTEClassifier(device=device)
    clf.fit(X_few_graph, y_few)

    y_pred = clf.predict(X_synth_graph)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(save_dir, "X_synth.npy"), X_synth)
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved inference results to {save_dir}")

if __name__ == "__main__":
    main()
