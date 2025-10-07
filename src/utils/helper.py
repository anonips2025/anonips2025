import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from scipy.stats import skewnorm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

def balanced_few_shot_sample(X, y, num_shot, seed):
    np.random.seed(seed)
    labels, counts = np.unique(y, return_counts=True)
    num_labels = len(labels)
    ex_label = num_shot // num_labels
    ex_last_label = num_shot - ((num_labels - 1) * ex_label)
    ex_per_label = [ex_label] * (num_labels - 1) + [ex_last_label]
    indices = []
    for i, label in enumerate(labels):
        label_indices = np.where(y == label)[0]
        chosen = np.random.choice(label_indices, ex_per_label[i], replace=False)
        indices.extend(chosen)
    np.random.shuffle(indices)
    return X[indices], y[indices]

def class_balanced_serialized_sample(data, n_shots, label_key='output', seed=None):
    """
    Perform class balanced sampling on the serialized data json.
    
    Parameters:
      data (list): List of dictionaries, each representing an instance.
      n_shots (int): Total number of samples desired.
      label_key (str): The key used for labels in each dictionary.
      seed (int, optional): Random seed for reproducibility.
    
    Returns:
      list: A list of sampled dictionaries, balanced across the unique classes.
    """
    if seed is not None:
        random.seed(seed)

    # Get unique classes from the data for the specified label_key
    unique_classes = list({d[label_key] for d in data})
    num_classes = len(unique_classes)
    
    # Desired samples per class (using floor division)
    shots_per_class = n_shots // num_classes
    sampled_by_class = {}
    
    # Sample for each class
    for cls in unique_classes:
        class_samples = [d for d in data if d[label_key] == cls]
        if len(class_samples) < shots_per_class:
            # Warn if there are not enough samples and take all available
            print(f"Warning: Not enough samples for class {cls}. "
                  f"Requested {shots_per_class}, but only found {len(class_samples)}.")
            sampled_by_class[cls] = class_samples
        else:
            sampled_by_class[cls] = random.sample(class_samples, shots_per_class)
    
    # Combine sampled instances
    sampled_data = []
    for cls in unique_classes:
        sampled_data.extend(sampled_by_class[cls])
    
    # If total samples are less than desired (due to shortage in one or more classes),
    # add extra samples from those classes that have extras.
    total_samples = len(sampled_data)
    if total_samples < n_shots:
        needed = n_shots - total_samples
        extras = []
        for cls in unique_classes:
            all_class_samples = [d for d in data if d[label_key] == cls]
            current_samples = sampled_by_class[cls]
            # Find remaining instances (preserving order is not crucial here)
            remaining = [d for d in all_class_samples if d not in current_samples]
            extras.extend(remaining)
        if len(extras) >= needed:
            sampled_data.extend(random.sample(extras, needed))
        else:
            sampled_data.extend(extras)
    
    # Shuffle final sampled data
    random.shuffle(sampled_data)
    return sampled_data

def get_few_shot_from_csv(dataset_name, num_shot, seed):
    """
    Loads the original dataset CSV, encodes the class column to binary labels, splits into train/test using stratified sampling,
    and returns balanced few-shot samples from the train set as pandas DataFrame/Series.
    Dynamically adjusts train size if needed to ensure enough samples for balanced few-shot sampling.
    Returns: X_few (DataFrame), y_few (Series), X_train (DataFrame), y_train (Series), X_test (DataFrame), y_test (Series)
    """

    np.random.seed(seed)
    # Load original dataset
    orig_path = f"dataset/{dataset_name}/{dataset_name}.csv"
    df = pd.read_csv(orig_path)
    if 'class' not in df.columns:
        raise ValueError("The original dataset must have a 'class' column.")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['class'].astype(str)).astype(bool)
    feature_cols = [col for col in df.columns if col not in ["class", "label"]]
    X = df[feature_cols]
    y = df["label"]
    

    labels, counts = np.unique(y, return_counts=True)
    num_classes = len(labels)
    shots_per_class = num_shot // num_classes

    shots_last_class = num_shot - ((num_classes - 1) * shots_per_class)
    max_shots_needed = max(shots_per_class, shots_last_class)
    
    initial_train_size = 0.4
    train_size = initial_train_size
    
    while train_size <= 0.9:
        expected_train_per_class = []
        for count in counts:
            expected_samples = int(count * train_size)
            expected_train_per_class.append(expected_samples)
        
        min_train_per_class = min(expected_train_per_class)
        

        if min_train_per_class >= max_shots_needed:
            break
            

        train_size += 0.1
    

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    

    X_train_np = X_train.values
    y_train_np = y_train.values
    X_few_np, y_few_np = balanced_few_shot_sample(X_train_np, y_train_np, num_shot, seed)

    X_few = pd.DataFrame(X_few_np, columns=feature_cols)
    y_few = pd.Series(y_few_np)
    return X_few, y_few, X_train, y_train, X_test, y_test