import os
import pickle
import numpy as np
import pandas as pd
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--numshot", type=int, required=True)
    parser.add_argument("--eval_res_dir", type=str, default="eval_res/tabllm")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    args = parser.parse_args()
    extract_tabllm_inference(args.dataset, args.numshot, args.eval_res_dir, args.dataset_dir)

if __name__ == "__main__":
    main()
