import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.ttnet.model import TTnet_general
from src.ttnet_rules.save_preprocess import prune_ttnet

def parse_info_file(info_path):
    categorical_indices = []
    continuous_indices = []
    with open(info_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.endswith('continuous'):
                continuous_indices.append(i)
            elif line.endswith('discrete'):
                categorical_indices.append(i)
    # Remove label column if present
    with open(info_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('class'):
                label_idx = idx
                if idx in categorical_indices: # Check if label_idx was indeed in categorical_indices
                    categorical_indices.remove(idx)
                if idx in continuous_indices: # Check if label_idx was indeed in continuous_indices
                    continuous_indices.remove(idx)
                # categorical_indices = [i for i in categorical_indices if i != label_idx] # Original logic
                # continuous_indices = [i for i in continuous_indices if i != label_idx] # Original logic
                break
    return categorical_indices, continuous_indices

class TTNetPreprocessor:
    def __init__(self, info_path):
        self.info_path = info_path
        self.categorical_indices, self.continuous_indices = parse_info_file(info_path)
        self.ohe = None
        self.scaler = None
        self.fitted = False

    def fit(self, X_fit):
        X_fit = np.array(X_fit)
        if len(self.categorical_indices) > 0:
            # Ensure there's data for categorical features if any are specified
            if X_fit.shape[1] <= max(self.categorical_indices, default=-1):
                # This case should ideally not happen if X_fit is valid and info_path matches X_fit structure
                print(f"Warning: Max categorical index {max(self.categorical_indices)} is out of bounds for X_fit with shape {X_fit.shape}. Skipping OHE fitting for these columns.")
                self.ohe = None # Or handle error
            else:
                self.ohe = OneHotEncoder(sparse_output=False, categories="auto", drop="first", handle_unknown='ignore')
                self.ohe.fit(X_fit[:, self.categorical_indices])
        
        if len(self.continuous_indices) > 0:
            if X_fit.shape[1] <= max(self.continuous_indices, default=-1):
                print(f"Warning: Max continuous index {max(self.continuous_indices)} is out of bounds for X_fit with shape {X_fit.shape}. Skipping Scaler fitting for these columns.")
                self.scaler = None # Or handle error
            else:
                self.scaler = StandardScaler()
                self.scaler.fit(X_fit[:, self.continuous_indices].astype(float))
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        X = np.array(X)
        
        # Handle categorical features
        if self.ohe is not None and len(self.categorical_indices) > 0:
             # Ensure X has enough columns for categorical_indices
            if X.shape[1] > max(self.categorical_indices, default=-1):
                X_cat = self.ohe.transform(X[:, self.categorical_indices])
            else: # Not enough columns in X for the stored categorical_indices
                print(f"Warning: X with shape {X.shape} has insufficient columns for categorical indices. Producing zero array for categorical part.")
                num_ohe_features = sum(len(cats) -1 for cats in self.ohe.categories_) if self.ohe.categories_ else 0
                X_cat = np.zeros((X.shape[0], num_ohe_features))
        else: # No OHE or no categorical indices
            X_cat = np.zeros((X.shape[0], 0))

        # Handle continuous features
        if self.scaler is not None and len(self.continuous_indices) > 0:
            if X.shape[1] > max(self.continuous_indices, default=-1):
                X_cont = self.scaler.transform(X[:, self.continuous_indices].astype(float))
            else: # Not enough columns for continuous
                print(f"Warning: X with shape {X.shape} has insufficient columns for continuous indices. Producing zero array for continuous part.")
                num_cont_features = len(self.continuous_indices)
                X_cont = np.zeros((X.shape[0], num_cont_features))
        else: # No Scaler or no continuous_indices
            X_cont = np.zeros((X.shape[0], 0))
            
        # Concatenate features - handle edge cases
        if X_cat.shape[1] > 0 and X_cont.shape[1] > 0:
            X_proc = np.concatenate([X_cat, X_cont], axis=1)
        elif X_cat.shape[1] > 0:
            X_proc = X_cat
        elif X_cont.shape[1] > 0:
            X_proc = X_cont
        else:
            # No features at all - create a dummy feature
            X_proc = np.zeros((X.shape[0], 1))
            
        index = X_cat.shape[1] # index where continuous features start
        return X_proc.astype(np.float32), index

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class TTnetStudentModel:
    def __init__(self, features_size, index, device='cpu', **kwargs):
        self.device = device
        self.model = TTnet_general(features_size=features_size, index=index, device=device, **kwargs).to(device)
        self.trained = False
    def fit(self, X, y, epochs=30, batch_size=128, lr=1e-3):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(X_tensor.size(0))
            for i in range(0, X_tensor.size(0), batch_size):
                idx = perm[i:i+batch_size]
                xb, yb = X_tensor[idx], y_tensor[idx]
                optimizer.zero_grad()
                out = self.model(xb)
                loss = torch.nn.BCEWithLogitsLoss()(out, yb)
                loss.backward()
                optimizer.step()
        self.trained = True

        # Prune the model after training
        # Ensure X and y are numpy arrays for prune_ttnet
        X_np = X if isinstance(X, np.ndarray) else np.array(X)
        y_np = y if isinstance(y, np.ndarray) else np.array(y)
        pruned_model, prune_info = prune_ttnet(self.model, X_np, y_np)

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs
    def extract_rules(self):
        """
        Extract DNF rules from the TTNet model after pruning, and count the number of gates.
        Returns:
            dict: { 'dnfs': [...], 'number_of_gates': int }
        """
        try:
            from src.ttnet_rules.utils import convert_to_tt
            # Convert TTNet block to truth table and DNF (in memory)
            convert_LTT = convert_to_tt(self.model, path_save_model=None)
            if not convert_LTT or 'dnfs' not in convert_LTT:
                raise ValueError("Failed to extract DNF rules from TTNet")
            dnfs = convert_LTT['dnfs']
            # Count the number of gates (sum of '&' and '|' in all DNF strings)
            number_of_gates = sum(dnf.count('&') + dnf.count('|') for dnf in dnfs)
            number_of_rules = np.count_nonzero(self.model.classifier.out.weight.data.cpu().numpy())
            return {'dnfs': dnfs, 'complexity': number_of_gates+number_of_rules}
        except Exception as e:
            print(f"Error extracting rules from TTNet: {str(e)}")
            return None
