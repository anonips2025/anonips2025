import ast
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils import prune

from src.ttnet.classifier import Classifier, ClassifierBNN


class LinearToPrune(nn.Module):
    def __init__(self, features_size_LR=100, nclass=2):
        super(LinearToPrune, self).__init__()
        self.nclass = nclass
        self.out = nn.Linear(
            features_size_LR, 1 if nclass == 2 else nclass
        )  # Linear layer with output size 1 for binary classification

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # Flatten the input tensor
        x = self.out(x)  # Apply the linear layer
        if self.nclass <= 2:
            x = torch.sigmoid(x)  # Apply sigmoid activation for binary classification
        else:
            x = F.softmax(
                x, dim=1
            )  # Apply softmax activation for multi-class classification
        return x


def model_to_numpy(classifier_model):
    """
    Converts the classifier model's weights and batch normalization parameters to numpy arrays.

    Args:
        classifier_model (nn.Module): The classifier model containing the layers to be converted.

    Returns:
        tuple: A tuple containing the weights, biases, scale, and bias parameters in numpy format.
    """
    def _get_scale_bias(batch_norm, var, mean):
        std = torch.sqrt(var + batch_norm.eps)
        scale = batch_norm.weight / std
        bias = batch_norm.bias - mean * scale
        return scale.detach().cpu().numpy(), bias.detach().cpu().numpy()

    # Extract running statistics from the batch normalization layer
    var = classifier_model.classifier.BN1.running_var
    mean = classifier_model.classifier.BN1.running_mean
    scale, bias = _get_scale_bias(classifier_model.classifier.BN1, var, mean)

    # Check if the classifier is an instance of Classifier or ClassifierBNN and extract weights and biases accordingly
    if isinstance(classifier_model.classifier, Classifier):
        w1 = classifier_model.classifier.dense1.weight.detach().cpu().numpy()
        b1 = classifier_model.classifier.dense1.bias
        if b1 is None:
            b1 = np.zeros((w1.shape[0]))
        w2 = classifier_model.classifier.output.weight.detach().cpu().numpy()
        b2 = classifier_model.classifier.output.bias
        if b2 is None:
            b2 = np.zeros((w2.shape[0]))
    elif isinstance(classifier_model.classifier, ClassifierBNN):
        w1 = classifier_model.classifier.dense1.weight_bin.detach().cpu().numpy()
        b1 = classifier_model.classifier.dense1.bias
        if b1 is None:
            b1 = np.zeros((w1.shape[0]))
        w2 = classifier_model.classifier.output.weight_bin.detach().cpu().numpy()
        b2 = classifier_model.classifier.output.bias
        if b2 is None:
            b2 = np.zeros((w2.shape[0]))

    # Broadcast the scale to match the shape of the first layer's weights
    scale_broadcast = np.tile(scale.reshape(scale.shape[0], 1), (1, w1.shape[1]))

    return [w1, b1], [w2, b2], scale_broadcast, scale, bias


def prune_ttnet(model, x_train, y_train, mode="min_prune_percent", user_input=0):
    """
    Prunes the model based on the specified mode.

    Args:
        model: The neural network model to prune.
        x_train: Training data features (numpy array).
        y_train: Training data labels (numpy array).
        mode: "min_prune_percent" or "max_auc_tradeoff".
        user_input: Minimum prune percentage or maximum AUC tradeoff.

    Returns:
        pruned_model: The pruned model.
        prune_info: A dictionary containing pruning information.
    """
    if mode not in ["min_prune_percent", "max_auc_tradeoff"]:
        raise ValueError("Mode must be 'min_prune_percent' or 'max_auc_tradeoff'")

    prune_perc = np.linspace(0, 1, 21)  # Pruning percentages: 0%, 5%, ..., 100%
    auc_list = {}
    max_auc = -float("inf")
    best_prune = 0.0
    best_auc = -float("inf")

    original_device = next(model.parameters()).device
    pruned_model = model  # Work with the original model directly


    W1, W2, scale_broadcast, scale, bias = model_to_numpy(model)
    w1, b1 = W1
    w2, b2 = W2
    W = w2 @ (scale_broadcast * w1)
    B = w2 @ (scale * b1 + bias) + b2

    print(f"Sparsity for first layer: {np.sum(w1 == 0) / w1.size * 100}%")
    print(f"Sparsity for second layer: {np.sum(w2 == 0) / w2.size * 100}%")
    print(f"Sparsity equivalent layer: {np.sum(W == 0) / W.size * 100}%")

    x_train = torch.Tensor(x_train).to(original_device)
    n_classes = len(np.unique(y_train))

    # Store original classifier
    original_classifier = model.classifier
    best_classifier_state = None

    for i in prune_perc:
        # Skip pruning levels below the minimum prune percentage for "min_prune_percent" mode
        if mode == "min_prune_percent" and i < user_input:
            continue

        # Create a temporary classifier layer with weights and biases from the model
        temp_classifier = LinearToPrune(W.shape[1], W.shape[0])
        temp_classifier.out.weight = nn.Parameter(torch.Tensor(W.copy()))
        temp_classifier.out.bias = nn.Parameter(torch.Tensor(B.copy()))
        temp_classifier.to(original_device)

        parameters_to_prune = ((temp_classifier.out, "weight"),)

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=i,
        )

        # Temporarily replace the classifier for evaluation
        model.classifier = temp_classifier
        
        # Evaluate AUC for the pruned model
        with torch.no_grad():
            pred_proba = model(x_train).detach().cpu().numpy()

        # Calculate AUC score based on number of classes
        if n_classes <= 2:
            auc_score = roc_auc_score(y_train, pred_proba)
        else:
            # Use one-vs-rest AUC for multi-class
            auc_score = roc_auc_score(y_train, pred_proba, multi_class="ovr")

        auc_list[i] = auc_score

        if auc_score > max_auc:
            max_auc = auc_score

        if mode == "min_prune_percent":
            # Update best_prune only if the current prune level results in a better AUC
            if auc_score >= max_auc:
                best_prune = i
                best_auc = auc_score
                # Store the best classifier state
                best_classifier_state = temp_classifier.state_dict().copy()

        elif mode == "max_auc_tradeoff":
            auc_tradeoff = user_input
            # Find the largest pruning amount where the AUC is within the acceptable tradeoff
            if auc_score >= max_auc - auc_tradeoff:
                best_prune = i
                best_auc = auc_score
                # Store the best classifier state
                best_classifier_state = temp_classifier.state_dict().copy()

    # Apply the best pruning level to the model's classifier
    final_classifier = LinearToPrune(W.shape[1], W.shape[0])
    final_classifier.out.weight = nn.Parameter(torch.Tensor(W.copy()))
    final_classifier.out.bias = nn.Parameter(torch.Tensor(B.copy()))
    final_classifier.to(original_device)

    parameters_to_prune = ((final_classifier.out, "weight"),)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=best_prune,
    )

    # Replace the model's classifier with the pruned one
    model.classifier = final_classifier

    prune_info = {
        "mode": mode,
        "user_input": user_input,
        "auc_scores": list(auc_list.values()),
        "best_prune_amount": best_prune,
        "max_auc": max_auc,
        "best_auc": best_auc,
    }

    return model, prune_info


# Example usage:
# Assume ttnet is your trained model, and X_train, y_train are your training dataset
# mode = "min_prune_percent"
# user_input = 0.1  # Minimum prune percentage
# pruned_model, info = prune_ttnet(ttnet, X_train, y_train, mode, user_input)

# mode = "max_auc_tradeoff"
# user_input = 0.005  # Maximum AUC tradeoff
# pruned_model, info = prune_ttnet(ttnet, X_train, y_train, mode, user_input)


def save_thrs(model, repeat=3, save_path=None):
    preprocess = model.preprocess0
    classifier = model.classifier
    cont_features_number = preprocess.index1

    try:
        thr_bn = get_threshold_preprocess(preprocess, save_path)
    except:
        thr_bn = []

    return save_threshold_preprocess(
        thr_bn, repeat, cont_features_number, classifier, save_path
    )


def save_threshold_preprocess(
    thrs_bn, repeat, cont_features_number, classifier, directory=None
):
    try:
        classifier.Polynome_ACT
        poly_act = True
        poly_parameters = [
            classifier.Polynome_ACT.alpha,
            classifier.Polynome_ACT.beta,
            classifier.Polynome_ACT.gamma,
        ]
    except AttributeError:
        poly_act = False
        poly_parameters = []

    param = {
        "thresholds": thrs_bn,
        "repeat": repeat,
        "continous_features": cont_features_number,
        "poly_act": poly_act,
        "poly_parameters": poly_parameters,
    }

    return param


def get_threshold_preprocess(preprocess_model, path_save_model=None):
    """
    Collects statistics from the batch normalization layers of the preprocess model,
    computes thresholds, and returns them in memory.
    """
    def _get_scale_bias(batch_norm, var, mean):
        std = torch.sqrt(var + batch_norm.eps)
        scale = batch_norm.weight / std
        bias = batch_norm.bias - mean * scale
        return scale.detach().cpu().numpy(), bias.detach().cpu().numpy()
    with torch.no_grad():
        # Extract running statistics from the batch normalization layers
        var_BN0 = preprocess_model.BN0.running_var
        mean_BN0 = preprocess_model.BN0.running_mean
        scale_BN0, bias_BN0 = _get_scale_bias(preprocess_model.BN0, var_BN0, mean_BN0)

        var_BN1 = preprocess_model.BN1.running_var
        mean_BN1 = preprocess_model.BN1.running_mean
        scale_BN1, bias_BN1 = _get_scale_bias(preprocess_model.BN1, var_BN1, mean_BN1)

        var_BN2 = preprocess_model.BN2.running_var
        mean_BN2 = preprocess_model.BN2.running_mean
        scale_BN2, bias_BN2 = _get_scale_bias(preprocess_model.BN2, var_BN2, mean_BN2)

    # Compute thresholds
    thrs_bn = []
    # Compute threshold for BN0
    delta = bias_BN0[0]
    alpha = scale_BN0[0]
    thrs_bn.append(-delta / alpha)
    # Compute threshold for BN1
    delta = bias_BN1[0]
    alpha = scale_BN1[0]
    thrs_bn.append(-delta / alpha)
    # Compute threshold for BN2
    delta = bias_BN2[0]
    alpha = scale_BN2[0]
    thrs_bn.append(-delta / alpha)

    # Return thresholds, scales, and biases in memory
    bn_params = {
        "thresholds": thrs_bn,
        "scales": [scale_BN0, scale_BN1, scale_BN2],
        "biases": [bias_BN0, bias_BN1, bias_BN2],
    }
    return bn_params
