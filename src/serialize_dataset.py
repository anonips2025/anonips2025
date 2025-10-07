import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helper import get_few_shot_from_csv
from tabllm.external_datasets_variables import (
    template_config_bank, template_bank,
    template_config_blood, template_blood,
    template_config_calhousing, template_calhousing,
    template_config_creditg, template_creditg,
    template_config_diabetes, template_diabetes,
    template_config_heart, template_heart,
    template_config_income, template_income,
    template_config_jungle, template_jungle,
    template_config_albert, template_albert,
    template_config_compas, template_compas,
    template_config_covertype, template_covertype,
    template_config_credit_card_default, template_credit_card_default,
    template_config_electricity, template_electricity,
    template_config_eye_movements, template_eye_movements,
    template_config_road_safety, template_road_safety,
)


def serialize_single_file(dataset_name: str, df: pd.DataFrame, config, template, output_path: Path):
    """Serialize a DataFrame using a given config and template"""
    # Encode 'class' column using LabelEncoder
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['class'].astype(str)).astype(bool)

    def preprocess_row(row):
        processed = {}
        for col, value in row.items():
            if col in ['class', 'label']:
                continue

            actual_col = col
            if 'column_mapping' in config and col in config['column_mapping']:
                actual_col = config['column_mapping'][col]

            if 'pre' in config and col in config['pre']:
                try:
                    processed[actual_col] = config['pre'][col](value)
                except (KeyError, ValueError, TypeError) as e:
                    processed[actual_col] = str(value)
                    print(f"Warning: Preprocessing failed for {col}={value} in {dataset_name}: {str(e)}")
            else:
                processed[actual_col] = str(value)

            if 'post' in config:
                for new_col, func in config['post'].items():
                    processed[new_col] = func(row)
        return processed

    # Preprocess
    preprocessed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing {output_path.name}"):
        preprocessed.append(preprocess_row(row))

    # Serialize
    sentences = []
    for i, item in tqdm(enumerate(preprocessed), total=len(preprocessed), desc=f"Generating sentences for {output_path.name}"):
        try:
            sentence = template.format(**item)
            sentences.append({
                'note': sentence,
                'label': bool(df.iloc[i]['label'])
            })
        except KeyError as e:
            print(f"Missing key {e} in template for {output_path.name}")
            break

    # Save as HuggingFace dataset
    dataset = Dataset.from_pandas(pd.DataFrame(sentences))
    dataset.save_to_disk(output_path)
    print(f"Saved {output_path.name} with {len(sentences)} examples")


def serialize_dataset(dataset_name: str, seed: int = 0):
    config_map = {
        'bank': (template_config_bank, template_bank),
        'blood': (template_config_blood, template_blood),
        'calhousing': (template_config_calhousing, template_calhousing),
        'creditg': (template_config_creditg, template_creditg),
        'diabetes': (template_config_diabetes, template_diabetes),
        'heart': (template_config_heart, template_heart),
        'income': (template_config_income, template_income),
        'jungle': (template_config_jungle, template_jungle),
        'albert': (template_config_albert, template_albert),
        'compas': (template_config_compas, template_compas),
        'covertype': (template_config_covertype, template_covertype),
        'credit_card_default': (template_config_credit_card_default, template_credit_card_default),
        'electricity': (template_config_electricity, template_electricity),
        'eye_movements': (template_config_eye_movements, template_eye_movements),
        'road_safety': (template_config_road_safety, template_road_safety),
    }

    config, template = config_map[dataset_name]
    base_path = Path(f'dataset/{dataset_name}')
    output_base = Path('tabllm/datasets_serialized')

    # Get few-shot and test data using the same function as tabpfn_eval
    try:
        X_few, y_few, X_train, y_train, X_test, y_test = get_few_shot_from_csv(dataset_name, num_shot=4, seed=seed)
        
        # Combine test features and labels back into a DataFrame with 'class' column
        test_df = X_test.copy()
        test_df['class'] = y_test.values
        
        # Serialize test dataset
        serialize_single_file(dataset_name, test_df, config, template, output_base / f'{dataset_name}_test')
        print(f"Serialized test data for {dataset_name} with {len(test_df)} examples")
        
    except Exception as e:
        print(f"Error processing test data for {dataset_name}: {str(e)}")

    # Process synthetic dataset
    synth_path = base_path / f'{dataset_name}_synthetic.csv'
    if synth_path.exists():
        synth_df = pd.read_csv(synth_path)
        serialize_single_file(dataset_name, synth_df, config, template, output_base / f'{dataset_name}_synthetic')
    else:
        print(f"Synthetic dataset not found for {dataset_name}")


if __name__ == "__main__":
    seed = 0  # Default seed as requested
    for dataset in ['albert', 'bank', 'blood', 'calhousing', 'compas', 'covertype',
                   'credit_card_default', 'creditg', 'diabetes', 'electricity',
                   'eye_movements', 'heart', 'income', 'jungle', 'road_safety']:
        serialize_dataset(dataset, seed=seed)
