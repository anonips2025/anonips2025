import os

import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sympy import SOPform, symbols, to_cnf
from tqdm import tqdm


def get_exp_with_y(exp_DNFstr, exp_CNFstr):
    exp_DNFstr, exp_CNFstr = (
        str(exp_DNFstr).replace(" ", ""),
        str(exp_CNFstr).replace(" ", ""),
    )
    masks = exp_DNFstr.split("|")
    clausesnv = []
    for mask in masks:
        # print(mask)
        masknv = mask.replace("&", " | ")
        masknv = masknv.replace("x", "~x")
        masknv = masknv.replace("~~", "")
        masknv = masknv.replace(")", "").replace("(", "")
        masknv = "(" + masknv + ")"
        masknv = masknv.replace("(", "(y | ")
        clausesnv.append(masknv)
        # print(masknv)
    clauses = exp_CNFstr.split("&")
    for clause in clauses:
        # print(clause)
        clausenv = clause.replace("|", " | ")
        clausenv = clausenv.replace(")", "").replace("(", "")
        clausenv = "(" + clausenv + ")"
        clausenv = clausenv.replace(")", " | ~y)")
        clausesnv.append(clausenv)
    exp_CNF3 = " & ".join(clausesnv)

    return exp_CNF3


def get_expresion_methode1(n, condtion_filter, dc_filter=None):
    if dc_filter is not None:
        dc_filtervf = dc_filter
        condtion_filter_vf = [x for x in condtion_filter if x not in dc_filtervf]
        # print(len(condtion_filter_vf), len(dc_filtervf), len(dc_filtervf) / 2 ** self.n)
    else:
        condtion_filter_vf = condtion_filter

    if n == 4:
        w1, x1, y1, v1 = symbols("x_0, x_1, x_2, x_3")
        exp_DNF = SOPform(
            [w1, x1, y1, v1], minterms=condtion_filter_vf, dontcares=dc_filtervf
        )
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    elif n == 8:
        w1, x1, y1, v1, w2, x2, y2, v2 = symbols(
            "x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7"
        )
        exp_DNF = SOPform(
            [w1, x1, y1, v1, w2, x2, y2, v2],
            minterms=condtion_filter_vf,
            dontcares=dc_filtervf,
        )
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    elif n == 9:
        w1, x1, y1, v1, w2, x2, y2, v2, w3 = symbols(
            "x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8"
        )
        exp_DNF = SOPform(
            [w1, x1, y1, v1, w2, x2, y2, v2, w3],
            minterms=condtion_filter_vf,
            dontcares=dc_filtervf,
        )
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    elif n == 10:
        w1, x1, y1, v1, w2, x2, y2, v2, w3, x3 = symbols(
            "x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9"
        )
        exp_DNF = SOPform(
            [w1, x1, y1, v1, w2, x2, y2, v2, w3, x3],
            minterms=condtion_filter_vf,
            dontcares=dc_filtervf,
        )
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    # elif self.n == 5:
    #    w1, x1, y1, v1, w2, x2, y2, v2, w3, x3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9')
    #    exp_DNF = SOPform([w1, x1, y1, v1, w2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
    #    if self.with_contradiction:
    #        exp_CNF = POSform([w1, x1, y1, v1, w2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
    #    else:
    #        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    else:
        (
            w1,
            x1,
            y1,
            v1,
            w2,
            x2,
            y2,
            v2,
            w10,
            x10,
            y10,
            v10,
            w20,
            x20,
            y20,
            v20,
        ) = symbols(
            "x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7,x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15"
        )
        list_var = [
            w1,
            x1,
            y1,
            v1,
            w2,
            x2,
            y2,
            v2,
            w10,
            x10,
            y10,
            v10,
            w20,
            x20,
            y20,
            v20,
        ]
        exp_DNF = SOPform(
            list_var[:n], minterms=condtion_filter_vf, dontcares=dc_filtervf
        )
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    return exp_DNF, exp_CNF


def compute_corr_rules(tt_rules, n_filters, neurons_out, save_path, thr_corr):
    positive_correlation = []
    negative_correlation = []
    for xy_pixel in tqdm(range(neurons_out)):
        corr_matrix = np.zeros((n_filters, n_filters))
        for filteroccurence in range(n_filters):
            if xy_pixel in list(tt_rules[filteroccurence].keys()):
                values3 = tt_rules[filteroccurence][xy_pixel]
            else:
                values3 = None
            for filteroccurence2 in range(n_filters):
                if xy_pixel in list(tt_rules[filteroccurence2].keys()):
                    values3bis = tt_rules[filteroccurence2][xy_pixel]
                else:
                    values3bis = None
                if values3 is not None and values3bis is not None:
                    # print(values3, values3bis)
                    corr = np.sum(
                        1.0 * np.array(values3) == 1.0 * np.array(values3bis)
                    ) / len(values3bis)
                    # print(abs(corr - 1),  abs(corr))
                    if abs(corr - 1) > abs(corr):
                        corr_vf = corr - 1
                    else:
                        corr_vf = corr

                    # if filteroccurence == filteroccurence2:
                    #    assert corr ==1
                    flag_print = True

                else:
                    corr_vf = 0
                corr_matrix[filteroccurence, filteroccurence2] = corr_vf

                if corr_vf > thr_corr and filteroccurence != filteroccurence2:
                    if (
                        filteroccurence2,
                        filteroccurence,
                        xy_pixel,
                    ) not in positive_correlation:
                        positive_correlation.append(
                            (filteroccurence, filteroccurence2, xy_pixel)
                        )
                if corr_vf < -1 * thr_corr and filteroccurence != filteroccurence2:
                    if (
                        filteroccurence2,
                        filteroccurence,
                        xy_pixel,
                    ) not in negative_correlation:
                        negative_correlation.append(
                            (filteroccurence, filteroccurence2, xy_pixel)
                        )

    with open(os.path.join(save_path, "positive_correlation.txt"), "w") as f:
        f.write(str(positive_correlation))

    with open(os.path.join(save_path, "negative_correlation.txt"), "w") as f:
        f.write(str(negative_correlation))

    return positive_correlation, negative_correlation
