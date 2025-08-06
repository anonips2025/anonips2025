import ast
import json
import os
import random
from math import sqrt
from typing import Any, Dict

import mlflow
import numpy as np
import torch


def set_seed(seed):
    # Seed experiments
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_column_names(metadata):
    """
    Extracts the encoded column names from the provided metadata.

    Args:
        metadata (dict): A dictionary containing dataset metadata.

    Returns:
        list: A list of feature names (column names) from the dataset after TTnet encoding.
    """
    return metadata["dataset_shapes"]["ttnet"]["feature_names"]


def load_dnf(path_save_model, dnf_data=None):
    """
    Loads DNF expressions from memory if provided, otherwise from files (legacy).
    Args:
        path_save_model: (Unused if dnf_data is provided)
        dnf_data: Optional list of DNF expressions in memory.
    Returns:
        List of DNF expressions.
    """
    if dnf_data is not None:
        return dnf_data
    # Legacy fallback: load from files if needed
    dnf_files = os.listdir(path_save_model)
    dnf_files = [f for f in dnf_files if "DNF" in f]
    dnf_files.sort()
    return dnf_files


def pos_conv1D(shape, ksize=3, pad=0, stride=2):
    """Return the indices used in the 1D convolution"""

    arr = np.array(list(range(shape[0])))

    out = np.zeros(((arr.shape[0] - ksize + 2 * pad) // stride + 1, ksize))
    shape = out.shape

    for i in range(0, shape[0]):
        sub = arr[i * stride : i * stride + ksize]
        v = sub.flatten()
        out[i] = v

    return out.astype(int)


def load_thrs_human(metadata: Dict[str, Any], model, thrs, var_names):
    """
    Load thresholds for human-readable expressions and compute the indices used in the 1D convolution.

    Args:
        metadata (Dict[str, Any]): Metadata dictionary containing dataset shapes and encoding strategies.
        model: The model object containing convolutional layers.
        model_path (str): Path to the model directory.
        thrs: Thresholds and other parameters for processing.
        var_names (list): List of variable names.

    Returns:
        tuple: Updated variable names, feature name encoding dictionary, and indices used in the 1D convolution.
    """
    # Compute the shape of the input data after considering continuous features and repetitions
    shape = (
        metadata["dataset_shapes"]["ttnet"]["X"]["columns"]
        + thrs["continous_features"]
        + (thrs["repeat"]) * (abs(thrs["continous_features"])),
    )

    mlflow.log_param("shape", shape)

    # Extract convolution parameters from the model
    kernel_size = model.block.conv1.kernel_size[0]
    stride = model.block.conv1.stride[0]
    padding = model.block.conv1.padding[0]

    # Compute the indices used in the 1D convolution
    indexes = pos_conv1D(shape, kernel_size, padding, stride)

    human_thresh = []
    # Check if there are continuous features to process
    if (
        metadata["dataset_shapes"]["ttnet"]["X"]["columns"] + thrs["continous_features"]
        > 0
    ):
        # Separate continuous feature names from the variable names
        cont_feat_names = var_names[thrs["continous_features"] :]
        var_names = var_names[: thrs["continous_features"]]

        # Extract mean and standard deviation of continuous features encoded with standard normalization
        continuous_features = [
            feature
            for feature, params in metadata["encoding_strategy"]["ttnet"][
                "features"
            ].items()
            if isinstance(params, dict) and params.get("type") == "standard"
        ]
        mean = [
            metadata["encoding_strategy"]["ttnet"]["features"][feature][
                "encoder_params"
            ]["mean"]
            for feature in continuous_features
        ]
        std = [
            sqrt(
                metadata["encoding_strategy"]["ttnet"]["features"][feature][
                    "encoder_params"
                ]["var"]
            )
            for feature in continuous_features
        ]

        # Compute thresholds for each repeat and update continuous feature names with its thresholds
        for r in range(thrs["repeat"]):
            scale = thrs["thresholds"]["scales"][r]
            bias = thrs["thresholds"]["biases"][r]
            thresholds = mean - std * bias / scale
            human_thresh.append(thresholds)
            cont_feat_names_thr = [
                f"{cont_feat_names[i]} >= {round(thresholds[i], 2)}"
                for i in range(len(cont_feat_names))
            ]
            var_names = var_names + cont_feat_names_thr

    # Create a dictionary to encode feature names with their indices
    fname_encode = {i: val for i, val in enumerate(var_names)}

    return var_names, fname_encode, indexes


def get_human_expr(dnf, patch):
    # Initialize an empty list to store the final expression
    e = []

    # Iterate over each conjunction (AND conditions) in the disjunctive normal form (DNF) clause
    for and_conds in dnf:
        # Initialize an empty list to store individual conditions
        cond = []
        # Iterate over each condition index in the conjunction
        for and_idx in and_conds:
            # Extract the index from the condition string (e.g., 'x_1' -> 1)
            idx = int(and_idx.split("_")[-1])
            # Determine if the condition is negated (starts with '~')
            if and_idx.startswith("~"):
                if " >= " in patch[idx]:
                    # If it's a continuous feature, change ' >= ' to ' < '
                    cond.append(patch[idx].replace(" >= ", " < "))
                else:
                    # Else, it's a discrete feature, change ' = ' to ' != '
                    cond.append(patch[idx].replace(" == ", " != "))
            else:
                # Append the condition to the list with the corresponding human-readable feature name (patch[idx])
                cond.append(patch[idx])
        # Join all conditions in the conjunction with ' & ' (AND) operator
        cond = " & ".join(cond)
        # Append the conjunction to the final expression list
        e.append(f"({cond})")
    # Join all conjunctions in the final expression list with ' | ' (OR) operator
    e = " | ".join(e)
    # Return the final human-readable expression
    return e


def get_human_expr_from_block(dnf, patches):
    expr = []

    for patch in patches:
        expr.append(get_human_expr(dnf, patch))
    return expr


def load_all_expr(model_path, var_names, dnf_files, indexes, truth_tables=None):
    """
    Load all expressions from memory (if provided) or from files (legacy).
    Args:
        model_path (str): Path to the model directory (unused if truth_tables is provided).
        var_names (list): List of variable names.
        dnf_files (list): List of DNF expressions (in memory or filenames).
        indexes (np.ndarray): Array of indices of the data used in the 1D convolution.
        truth_tables (np.ndarray, optional): Truth tables in memory.
    Returns:
        np.ndarray: Array of human-readable expressions.
    """
    # Use in-memory truth_tables if provided
    if truth_tables is not None:
        vinputs = np.array(var_names)
        filter_input = vinputs[indexes]
        all_express = []
        for filter_idx in range(truth_tables.shape[1]):
            # dnf_files is a list of DNF expressions (strings)
            if len(dnf_files) > filter_idx:
                dnf_filter = dnf_files[filter_idx]
            else:
                all_express.append([str(False) for _ in range(len(filter_input))])
                continue
            # Process the DNF content to extract individual conditions
            dnf_filter = dnf_filter.replace("(", "").replace(")", "").replace(" ", "").split("|")
            for i in range(len(dnf_filter)):
                dnf_filter[i] = dnf_filter[i].split("&")
            all_express.append(get_human_expr_from_block(dnf_filter, filter_input))
        all_express = np.array(all_express)
        return all_express
    # Legacy fallback: load from files
    truth_tables = np.load(os.path.join(model_path, "truth_table.npy"))
    vinputs = np.array(var_names)
    filter_input = vinputs[indexes]
    all_express = []
    for filter_idx in range(truth_tables.shape[1]):
        dnf_filter = [f for f in dnf_files if f"filter_{filter_idx}_" in f]
        if len(dnf_filter) > 0:
            dnf_filter = dnf_filter[0]
        else:
            all_express.append([str(False) for _ in range(len(filter_input))])
            continue
        with open(os.path.join(model_path, dnf_filter), "r") as f:
            dnf_filter = f.read()
        dnf_filter = (
            dnf_filter.replace("(", "").replace(")", "").replace(" ", "").split("|")
        )
        for i in range(len(dnf_filter)):
            dnf_filter[i] = dnf_filter[i].split("&")
        all_express.append(get_human_expr_from_block(dnf_filter, filter_input))
    all_express = np.array(all_express)
    return all_express


def save_expressions_as_human(all_express, model, model_path, output_classes):
    # Load weights and biases
    W = model.classifier.out.weight.data.cpu().numpy()
    B = model.classifier.out.bias.data.cpu().numpy()

    rules = []
    expres = all_express.flatten().tolist()
    total_complexity = 0
    total_rules = 0

    # Iterate over each class index in weights
    for i, class_idx in enumerate(W):
        # Add the intercept for the class
        intercept_rule = {
            "r": "",  # No rule expression for intercept
            "w": {output_classes[i]: float(B[i])},  # Intercept weight with class name
        }
        rules.append(intercept_rule)

        rules_idx = np.where(class_idx != 0)[0]
        for idx in rules_idx:
            rule = {
                "r": expres[idx],  # Rule expression
                "w": {},  # Weights associated with the rule
            }
            # Add non-zero weights to the rule
            for j, weight in enumerate(W[:, idx]):
                if weight != 0:
                    rule["w"][output_classes[j]] = float(weight)
                    total_rules += 1
                    total_complexity += rule["r"].count("&") + rule["r"].count("|")
            rules.append(rule)

    # No file saving, just return the rules and expressions
    return {"rules": rules, "expressions": expres}


def human_donctcares_on_expr(
    save_path,
    block_occurence,
    fname_encode,
    input_var,
    filter_idx,
    shapeici_out,
    xy_pixel,
    all_expr,
    ksize,
):
    # print(input_var)
    variables = {
        i: fname_encode[input_var[i]].replace("?", "NoInfo")
        for i in range(len(input_var))
    }

    tt = np.load(os.path.join(save_path, "truth_table.npy"))
    tt = tt[:, filter_idx]
    nbits = int(np.log2(tt.shape[0]))
    W_LR = np.load(os.path.join(save_path, "W.npy"))

    binary_numbers = generate_binary_strings(nbits)

    df = pd.DataFrame(
        [list(string) for string in binary_numbers],
        columns=[f"{i}" for i in range(nbits)],
    )
    df[f"Filter_{filter_idx}_Value_1"] = tt.astype(bool)
    df[f"Filter_{filter_idx}_dontcares_1"] = tt.astype(bool) & False
    tt = df.copy()
    path_tt_dc = os.path.join(
        save_path,
        "human_expressions",
        "Truth_Table_block"
        + str(block_occurence)
        + "_filter_"
        + str(filter_idx)
        + "_"
        + str(xy_pixel)
        + ".csv",
    )

    names_features = list(variables.values())
    num = []
    separate_f = []
    separate_f_var = []
    for ix, x in enumerate(names_features):
        umici = x.split("_")[0]
        if umici not in num:
            num.append(umici)
            separate_f.append([x])
            separate_f_var.append(["x_" + str(ix)])
        else:
            for indexpos in range(len(num)):
                if num[indexpos] == umici:
                    separate_f[indexpos].append(x)
                    separate_f_var[indexpos].append("x_" + str(ix))
            # else:
        # print(separate_f_var)
        # print(separate_f, num, umici, umici not in num)
    # we compute the expressions of expressions with less than  ksize clauses
    # print(separate_f, num, separate_f_var)
    if len(separate_f) < ksize:
        # print("ok1")
        var_sum = sum([len(f) for f in separate_f])
        assert var_sum == len(input_var)
        all_values = []
        for j in range(len(separate_f)):
            bin_values = [
                [0] * len(separate_f[j]) for _ in range(len(separate_f[j]) + 1)
            ]
            # print("bin_values 0 ", bin_values)
            for i in range(len(separate_f[j])):
                bin_values[i + 1][i] = 1
            # print("bin_values 1 ",bin_values)
            all_values.append(bin_values)
        if len(all_values) == 1:
            table = all_values[0]
        else:
            table = list(itertools.product(all_values[0], all_values[1]))
            buff = []
            for ii, item in enumerate(table):
                buff.append(item[0] + item[1])
            del table
            table = copy.copy(buff)

        # print("all_values", all_values)
        # print("tbale 0", table)

        # print(ok)
        if len(all_values) > 2:
            for i in range(2, len(all_values)):
                # print("i, t, 0 ", i, table)
                table = list(itertools.product(table, all_values[i]))
                # print("i, t, 1 ", i, table)
                buff = []
                for ii, item in enumerate(table):
                    buff.append(item[0] + item[1])
                del table
                table = copy.copy(buff)
        # print("table 1", table)
        for tablevalue in table:
            # cpttabassert = 0
            # for tablevaluex in tablevalue:
            # cpttabassert+=len(tablevaluex)
            # print(cpttabassert,  args.kernel_size_per_block[block_occurence])
            assert len(tablevalue) == nbits
        # print("all_values", all_values)

        p = 1

        # print(table)
        # print(ok)

        for val in separate_f:
            p *= len(val) + 1

        assert len(table) == p

        small_tt = []
        small_tt2 = []
        for t in table:
            if type(t) is tuple:
                concat = np.concatenate(t)
            else:
                concat = t
            # print(concat)

            small_tt2.append(int("".join(str(i) for i in concat), 2))
            # if args.random_permut:
            #     separate_f_var2 = []
            #     for sfvar in separate_f_var:
            #         for sfvar2 in sfvar:
            #             separate_f_var2.append(sfvar2)
            #     concat_new = [0] * len(concat)
            #     for index_concat in range(len(concat)):
            #         valuecat = concat[index_concat]
            #         value_position = int(separate_f_var2[index_concat].replace("x_", ""))
            #         concat_new[value_position] = valuecat
            #     # print(concat_new)
            #     del concat
            #     concat = concat_new

            small_tt.append(int("".join(str(i) for i in concat), 2))

        # print(small_tt2)
        # print(small_tt)
        # print(ok)

        tt_dc = tt.copy()

        for idx in tt.index.values.tolist():
            # print(idx)
            # print(tt.iloc[idx])
            # print(tt.iloc[idx].drop(['Filter_0_Value_1',  'Filter_0_dontcares_1']).to_numpy())

            to_int = (
                tt.iloc[idx]
                .drop(
                    [f"Filter_{filter_idx}_Value_1", f"Filter_{filter_idx}_dontcares_1"]
                )
                .to_numpy()
            )
            value = int("".join(str(i) for i in to_int), 2)
            # if args.dc_logic:
            if value not in small_tt:
                tt_dc.at[idx, f"Filter_{filter_idx}_dontcares_1"] = True
        # print(tt0)
        for df2 in [tt_dc]:
            # print(df2)
            answer = df2[f"Filter_{filter_idx}_Value_1"].to_numpy()
            dc = df2[f"Filter_{filter_idx}_dontcares_1"].to_numpy()
            condtion_filter = df2.index.values[answer].tolist()
            dc_filter = df2.index.values[dc].tolist()
            # condtion_filter_cnf = df2["index"].values[answer_cnf].tolist()
            # print(condtion_filter, dc_filter)
            df2.to_csv(path_tt_dc)
            exp_DNF, exp_CNF = get_expresion_methode1(
                nbits, condtion_filter, dc_filter=dc_filter
            )
            # exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
            exp_CNF3 = get_exp_with_y(exp_DNF, exp_CNF)
            # print(exp_DNF, variables)

            # module.save_cnf_dnf(1.0, exp_CNF3, exp_DNF, exp_CNF, xypixel=xy_pixel)
            dnf = (
                str(exp_DNF)
                .replace(" ", "")
                .replace(")", "")
                .replace("(", "")
                .split("|")
            )
            dnf = [d.split("&") for d in dnf]
            if dnf != [["False"]] and dnf != [["True"]]:
                readable_dnf = get_human_expr(dnf, names_features)
            else:
                readable_dnf = dnf[0][0]
            # readable_dnf = readable_expr(exp_DNF, variables, args)
            # print(readable_dnf)
            # print(f"Position : {xy_pixel}")
            # print()
            # print("Weight : ")
            list_W = []
            for idx in range(W_LR.shape[0]):
                # print(W_LR[idx][filteroccurence * shapeici_out + xy_pixel].item())
                # print(W_LR[idx_negative][filteroccurence * shapeici_out + xy_pixel].item())
                # print()
                list_W.append(W_LR[idx][filter_idx * shapeici_out + xy_pixel].item())
            # all_expr.append(
            #    (readable_dnf,
            #     list_W))

    else:
        tt.to_csv(path_tt_dc)
        # module.save_cnf_dnf(1.0, exp_CNF3, exp_DNF, exp_CNF, xypixel=xy_pixel)
        variables = {
            i: fname_encode[input_var[i]].replace("?", "NoInfo")
            for i in range(len(input_var))
        }

        answer = df[f"Filter_{filter_idx}_Value_1"].to_numpy()
        dc = df[f"Filter_{filter_idx}_dontcares_1"].to_numpy()
        condtion_filter = df.index.values[answer].tolist()
        dc_filter = df.index.values[dc].tolist()

        exp_DNF, exp_CNF = get_expresion_methode1(
            nbits, condtion_filter, dc_filter=dc_filter
        )
        # exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
        exp_CNF3 = get_exp_with_y(exp_DNF, exp_CNF)
        dnf = str(exp_DNF).replace(" ", "").replace(")", "").replace("(", "").split("|")

        if not ("None" in dnf or "False" in dnf or "True" in dnf):
            dnf = [d.split("&") for d in dnf]

            readable_dnf = get_human_expr(dnf, names_features)
        else:
            readable_dnf = dnf

        # print(exp_DNF, variables)
        # readable_dnf = readable_expr(exp_DNF, variables, args)
        # print(readable_dnf)
        # print(f"Position : {xy_pixel}")
        # print()
        # print("Weight : ")
        list_W = []
        for idx in range(W_LR.shape[0]):
            # print(W_LR[idx][filteroccurence * shapeici_out + xy_pixel].item())
            # print(W_LR[idx_negative][filteroccurence * shapeici_out + xy_pixel].item())
            # print()
            list_W.append(W_LR[idx][filter_idx * shapeici_out + xy_pixel].item())
        # all_expr.append(
        #    (readable_dnf,
        #     list_W))
        # print()

    all_expr.append((readable_dnf, list_W))

    return exp_CNF3, exp_DNF, exp_CNF, all_expr, readable_dnf


def inject_dc_terms_on_network(vinputs, indexes, path_save_model, fname_encode, ksize):
    truth_tables = np.load(os.path.join(path_save_model, "truth_table.npy"))
    W = np.load(os.path.join(path_save_model, "W.npy"))
    n_filters = truth_tables.shape[-1]
    neurons_per_filter = W.shape[1] // (n_filters)

    print("Injecting dont care terms ...")

    reverse_dict = {value: key for key, value in fname_encode.items()}

    all_expr = []
    for filter_idx in range(n_filters):
        filter_input = vinputs[indexes]
        for xy_pixel in range(neurons_per_filter):
            patch = filter_input[xy_pixel]

            patch = [reverse_dict[val] for val in patch]
            cnf3, dnf, cnf, all_expr, human_dnf = human_donctcares_on_expr(
                path_save_model,
                0,
                fname_encode,
                patch,
                filter_idx,
                neurons_per_filter,
                xy_pixel,
                all_expr,
                ksize,
            )

    return all_expr


def main(
    metadata: Dict[str, Any], model, model_path, dataset=None, seed=None, args=None
):
    if dataset is None:
        config_general = Config()
        dataset = config_general.dataset

    set_seed(seed)

    # for weight_decay in [None, 0.001, 0.01]:
    #     path_model = os.path.join(path_save_model, f'_{weight_decay}')

    var_names = get_column_names(metadata)
    with open(os.path.join(model_path, "thresholds_rules.json"), "r") as jfile:
        thrs = json.load(jfile)

    try:
        thrs["thresholds"] = ast.literal_eval(thrs["thresholds"])
    except ValueError:
        thrs["thresholds"] = eval(thrs["thresholds"])
    thrs["repeat"] = int(thrs["repeat"])
    thrs["continous_features"] = int(thrs["continous_features"])
    # except:
    #     thrs["continous_features"] = -1
    thrs["poly_parameters"] = ast.literal_eval(thrs["poly_parameters"])

    dnf_files = load_dnf(model_path)
    var_names, fname_encode, indexes = load_thrs_human(
        metadata, model, model_path, thrs, var_names
    )
    vinputs = np.array(var_names)
    ksize = args.kernel_size[0]
    all_express = load_all_expr(model_path, var_names, dnf_files, indexes)
    save_expressions_as_human(all_express, model_path)
    all_express_dc = inject_dc_terms_on_network(
        vinputs, indexes, path_save_model, fname_encode, ksize
    )
    nrules, complexity = save_expressions_with_dc(all_express_dc, path_save_model)
    compute_correlation_tt(path_save_model, dataset)

    return nrules, complexity


if __name__ == "__main__":
    dataset = "adult"
    seeds = [6, 7, 8]  # [0, 1, 6, 7, 8]
    for seed in seeds:
        main(dataset, seed)
