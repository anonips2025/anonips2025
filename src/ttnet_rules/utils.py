import os

import numpy as np
import torch
from torch.utils.data import TensorDataset


def convert_to_tt(ttnet, path_save_model=None):
    """
    Converts a TTNet model to truth tables and returns the results.

    Args:
        ttnet: The TTNet model to convert.
        path_save_model: (Unused) Kept for compatibility, but no longer used for saving.

    Returns:
        A dictionary containing the truth tables, the DNF expressions, the number of gates, and the number of nodes.
    """
    print(ttnet)
    # Convert the TTNet model to truth tables
    truth_tables = ttnet.convert()
    print("\n Truth Tables learned \n")
    print(truth_tables)
    print()

    # Initialize gate counter and DNF list
    cpt_gates = 0
    dnfs = []

    # Iterate over each filter occurrence in the truth tables
    for filter_occurence in range(truth_tables.shape[1]):
        # Get the CNF, DNF, and CNF3 expressions for the current filter
        exp_CNF, exp_DNF, exp_CNF3 = ttnet.block.get_TT_block_1filter(
            filterici=filter_occurence,
            path_save_exp=None,  # No file saving
        )

        # Print and count the gates in the DNF expression
        print(f"DNF number {filter_occurence + 1}: {exp_DNF}")
        cpt_gates += str(exp_DNF).count("&") + str(exp_DNF).count("|")
        print(f"Current gate count: {cpt_gates}\n")

        # Append the DNF expression to the dnfs list
        dnfs.append(f"DNF number {filter_occurence + 1}: {exp_DNF}")

    # Number of nodes after 1D CNN layer
    num_nodes = ttnet.block.conv1.out_channels if hasattr(ttnet.block, 'conv1') else truth_tables.shape[1]

    # Return the results as a dictionary
    return {"truth_tables": truth_tables, "dnfs": dnfs, "number_of_gates": cpt_gates, "num_nodes": num_nodes}
