import ast
import warnings
warnings.filterwarnings('ignore')
import random
import os
import json
import numpy as np
import pandas as pd
import torch
import re

from aix360.algorithms.rbm import FeatureBinarizer, LogisticRuleRegression


# Load configuration

seed = 0


def set_seed(seed):
    # Seed experiments
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rules_aix360_to_human(model, dataset_name, save_path, seed, data_root="./dataset"):

    file_name = os.path.join(save_path, 'rules.txt')
    all_rules = ''
    index_to_colnames = colnames_index_dict(dataset_name, os.path.join(data_root, dataset_name))
    info = read_info(f"{data_root}/{dataset_name}/{dataset_name}.info")
    with open(f"{data_root}/{dataset_name}/columns.json", "r") as f:
        col = json.load(f)
    colnames_dict = {}
    for f in info[0][:-1]:
        colnames_dict[int(f[0])] = col[int(f[0])-1]

    _,_,_,_,_,_,db_enc = load_data(dataset_name, ".", seed, private=False, num_shots="all", return_db_enc=True)

        # Function to replace numbers with column names
    def replace_columns(rules, column_map):
        updated_rules = []
        for rule in rules:
            # Split the rule into parts
            parts = rule.split()
            # Replace the numbers with the corresponding column names
            updated_parts = [
                column_map.get(part, part) for part in parts
            ]
            # Join the parts back into a single string
            updated_rule = " ".join(updated_parts)
            updated_rules.append(updated_rule)

        return updated_rules

    def replace_int(rules, mean, std):
        updated_rules = []
        for rule in rules:
            parts = rule.split()

            for i in range(len(parts)):
                if parts[i] in list(mean.index):
                    parts[i+2] = "{:.2f}".format(float(parts[i+2]) * std[parts[i]] + mean[parts[i]])
            updated_rule = " ".join(parts)
            updated_rules.append(updated_rule)
        return updated_rules
    
    def replace_column_names(rules, mapping):
        # Define operators and the pattern to split clauses while retaining logical connectors
        operators = ["==", "!=", ">", "<", ">=", "<="]
        clause_pattern = r'(\sAND\s|\sOR\s)'
        
        updated_rules = []

        for rule in rules:
            # Split by logical connectors (AND, OR) to isolate each clause
            clauses = re.split(clause_pattern, rule)
            updated_clauses = []
            
            for clause in clauses:
                # Skip connectors (AND/OR)
                if clause.strip() in ["AND", "OR"]:
                    updated_clauses.append(clause)
                    continue
                
                # Split each clause by operators to identify left side and right side
                parts = re.split(r'(\s==\s|\s!=\s|\s>\s|\s<\s|\s>=\s|\s<=\s)', clause)
                if len(parts) < 3:
                    # Add clause as-is if it doesn't match the expected pattern
                    updated_clauses.append(clause)
                    continue

                # Extract left part (condition side), operator, and right part (value side)
                left_part, operator, right_part = parts[0].strip(), parts[1].strip(), parts[2].strip()
                
                # Check if left part contains an underscore or is a standalone integer
                if "_" in left_part:
                    prefix_str = left_part.split("_")[0]
                    if prefix_str.isdigit():
                        prefix = int(prefix_str)
                        if prefix in mapping:
                            suffix = left_part[left_part.index('_'):]  # Keep the suffix after "_"
                            left_part = mapping[prefix] + suffix
                elif left_part.isdigit():
                    prefix = int(left_part)
                    if prefix in mapping:
                        left_part = mapping[prefix]

                # Reassemble the updated clause
                updated_clause = f"{left_part} {operator} {right_part}"
                updated_clauses.append(updated_clause)
            
            # Join all updated clauses and connectors to form the final rule
            updated_rule = "".join(updated_clauses)
            updated_rules.append(updated_rule)
        
        return updated_rules

    if type(model) == LogisticRuleRegression:
        all_rules = model.explain()["rule"]
        coefficients = model.explain()["coefficient"]

        rules_with_colnames = replace_columns(all_rules, index_to_colnames)
        rules_with_original_continuous = replace_int(rules_with_colnames, db_enc.mean, db_enc.std)
        clean_rules = replace_column_names(rules_with_original_continuous, colnames_dict)
        # temp_filename = 'temp.csv'
        # all_rules.to_csv(temp_filename)
        # coefficients.to_csv('temp_coeff.csv')

        all_rules = ''
        for i, rule in enumerate(clean_rules):
            all_rules += str(coefficients[i]) + '\t' + rule + '\n'
    else:
        print("Not an explainer in : [LogisticRuleRegression]")
    # print(all_rules)
    with open(file_name, 'w') as file:
        file.write(str(all_rules))

    return all_rules



def colnames_index_dict(dataset_name, data_path='./dataset'):
    with open(os.path.join(data_path, dataset_name + ".columns"), 'r') as col_file:
        columns = col_file.readlines()
        columns = [[c.split('\t')[0], c.split('\t')[1]] for c in columns]

    index_to_colnames = {k: v.replace('\n', ' ') for k, v in columns}

    return index_to_colnames


