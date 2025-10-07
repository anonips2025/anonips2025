import pandas as pd
from ctgan import CTGAN
from pathlib import Path
from tqdm import tqdm


def load_info(info_path):
    discrete_cols = []
    label_col = None

    with open(info_path, 'r') as f:
        lines = f.read().splitlines()

    for line in lines:
        if line.startswith('LABEL_POS'):
            continue
        col, dtype = line.split()
        
        if dtype == 'discrete':
            discrete_cols.append(col if not col.isdigit() else int(col) - 1)  # support both names and positions

    return discrete_cols, label_col


def get_column_names(csv_path, discrete_positions):
    df = pd.read_csv(csv_path, nrows=1)
    column_names = df.columns.tolist()
    discrete_names = [column_names[i] if isinstance(i, int) else i for i in discrete_positions]
    return discrete_names


def generate_synthetic_data(dataset_name, k):
    dataset_path = Path(f'dataset/{dataset_name}/{dataset_name}.csv')
    info_path = Path(f'dataset/{dataset_name}/{dataset_name}.info')
    out_path = Path(f'dataset/{dataset_name}/{dataset_name}_synthetic.csv')

    print(f"\n[INFO] Processing dataset: {dataset_name}")
    print(f"[INFO] Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)

    print(f"[INFO] Loading info from {info_path}")
    discrete_positions, _ = load_info(info_path)
    discrete_columns = get_column_names(dataset_path, discrete_positions)
    print(f"[INFO] Discrete columns: {discrete_columns}")

    print(f"[INFO] Initializing CTGAN model")
    model = CTGAN()

    print(f"[INFO] Fitting CTGAN model on {dataset_name}...")
    model.fit(df, discrete_columns=discrete_columns)
    print(f"[INFO] Model fitting complete.")

    print(f"[INFO] Generating {k} synthetic samples...")
    synthetic_df = model.sample(k)
    print(f"[INFO] Synthetic data generation complete.")

    print(f"[INFO] Saving synthetic data to {out_path}")
    synthetic_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Saved synthetic data with {k} rows to {out_path}")


if __name__ == '__main__':

    datasets = [
        'albert', 'bank', 'blood', 'calhousing', 'compas', 'covertype',
        'credit_card_default', 'creditg', 'diabetes', 'electricity',
        'eye_movements', 'heart', 'income', 'jungle', 'road_safety'
    ]

    k = 5000

    for dataset in tqdm(datasets, desc="Datasets"):
        try:
            generate_synthetic_data(dataset, k)
        except Exception as e:
            print(f"Failed to generate for {dataset}: {e}")
