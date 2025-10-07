import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sympy import POSform, SOPform, symbols
from sympy.logic.boolalg import to_cnf
from torch import nn

from src.ttnet.modules import Binarize01Act, get_exp_with_y

act = Binarize01Act


class LTT_LR_1D(nn.Module):
    def __init__(
        self,
        chanel_interest=1,
        kernel_size=8,
        device="cpu",
        c_a_ajouter=0,
        T=0,
        filter_size=32,
        padding=0,
        stride=5,
        dropoutLTT=0,
    ):
        super(LTT_LR_1D, self).__init__()
        self.out_float = None
        self.dropoutLTT = nn.Dropout1d(dropoutLTT)
        self.chanel_interest = chanel_interest
        self.k = kernel_size
        self.n = self.chanel_interest * self.k
        self.device = device
        self.c_a_ajouter = c_a_ajouter
        self.T = T
        self.s = stride
        self.p = padding
        self.filter_size = filter_size
        self.conv1 = nn.Conv1d(
            1, filter_size, kernel_size=self.k, stride=self.s, padding=self.p
        )
        self.BN_AE11 = nn.BatchNorm1d(filter_size)
        self.nonlin = act(self.T)

    def forward(self, X):
        x_conv = self.conv1(X)
        self.out_float = self.BN_AE11(self.dropoutLTT(x_conv))
        out = self.nonlin(self.out_float)
        return out

    def get_TT_block_all_filter(self):
        self.eval()
        with torch.no_grad():
            # Generate truth table of size 2^self.n (self.n = channels * kernel size)
            l = [
                [int(y) for y in format(x, "b").zfill(self.n)] for x in range(2**self.n)
            ]
            df = pd.DataFrame(l)
            self.df = df.reset_index()

            # Convert truth table to tensor, shape = (2^self.n, channels, kernel size)
            x_input_f2 = torch.Tensor(l).reshape(
                2**self.n, self.chanel_interest, self.k
            )

            # Add extra channels to the truth table tensor based on c_a_ajouter
            y = x_input_f2.detach().clone()
            padding = torch.autograd.Variable(y)
            for itera in range(self.c_a_ajouter):
                x_input_f2 = torch.cat(
                    (x_input_f2, padding), 1
                )  # .type(torch.ByteTensor)
            del padding

            # Convert truth table tensor to the same device as the model and make inference
            x_input_f2 = x_input_f2.to(self.conv1.weight.device)
            out = self.forward(x_input_f2)  # out.shape = (2^self.n, filter_size, 1)

            # Convert output tensor to numpy array and reshape it to (2^self.n, filter_size)
            self.res_numpy = out.squeeze(-1).squeeze(-1).detach().cpu().clone().numpy()

            # If the output tensor is 1D, reshape it to 2D
            if len(self.res_numpy.shape) == 1:
                self.res_numpy = self.res_numpy.reshape((self.res_numpy.shape[0], 1))

            # Get the pre-activation output tensor and convert it to numpy array and reshape it to (2^self.n, filter_size)
            self.res_numpy_pre_activ = (
                self.out_float.squeeze(-1).squeeze(-1).detach().cpu().clone().numpy()
            )

            # Create a tensor of dontcares based on the pre-activation output tensor where values are 1 if the pre-activation output is greater than -T/2 and -1 if the pre-activation output is greater than T/2
            self.dontcares = 1.0 * (self.res_numpy_pre_activ > -self.T / 2) - 1.0 * (
                self.res_numpy_pre_activ > (self.T) / 2
            )

            # If the dontcares tensor is 1D, reshape it to 2D
            if len(self.dontcares.shape) == 1:
                self.dontcares = self.dontcares.reshape((self.dontcares.shape[0], 1))

        # Return the output tensor of the truth table inference
        return self.res_numpy

    def get_TT_block_1filter(self, filterici, path_save_exp):
        self.filterici = filterici
        self.blockici = 0
        self.path_save_exp = path_save_exp
        resici = self.res_numpy[:, filterici]
        dontcaresici = self.dontcares[:, filterici]
        unique = np.unique(resici)
        if len(unique) == 1:
            # Only save if path_save_exp is not None
            if self.path_save_exp is not None:
                self.save_cnf_dnf(resici[0], str(resici[0]))
                table = np.chararray((2**self.n, 2**self.n), itemsize=3)
                table[:][:] = str(resici[0])
                np.save(
                    self.path_save_exp
                    + "table_outputblock_"
                    + str(self.blockici)
                    + "_filter_"
                    + str(self.filterici)
                    + "_value_"
                    + str(resici[0])
                    + "_coefdefault_"
                    + str(resici[0])
                    + ".npy",
                    table,
                )
            exp_CNF, exp_DNF, exp_CNF3 = None, None, None
        else:
            exp_CNF, exp_DNF, exp_CNF3 = self.iterate_over_filter(
                resici, unique, dontcaresici, save=(self.path_save_exp is not None)
            )
        return exp_CNF, exp_DNF, exp_CNF3

    def save_cnf_dnf(self, coef, exp_CNF3, exp_DNF=None, exp_CNF=None):
        if self.path_save_exp is None:
            return
        # exp_CNF3 = str(coef)
        with open(
            self.path_save_exp
            + "table_outputblock_"
            + str(0)
            + "_filter_"
            + str(self.filterici)
            + "_coefdefault_"
            + str(coef)
            + ".txt",
            "w",
        ) as f:
            f.write(str(exp_CNF3))
        if exp_CNF is not None:
            with open(
                self.path_save_exp
                + "CNF_expression_block"
                + str(self.blockici)
                + "_filter_"
                + str(self.filterici)
                + "_coefdefault_"
                + str(coef)
                + "_sousblock_"
                + str(self.sousblockici)
                + ".txt",
                "w",
            ) as f:
                f.write(str(exp_CNF))
            with open(
                self.path_save_exp
                + "DNF_expression_block"
                + str(self.blockici)
                + "_filter_"
                + str(self.filterici)
                + "_coefdefault_"
                + str(coef)
                + "_sousblock_"
                + str(self.sousblockici)
                + ".txt",
                "w",
            ) as f:
                f.write(str(exp_DNF))

    def iterate_over_filter(self, resici, unique, dontcaresici, save=True):
        global exp_CNF, exp_DNF, exp_CNF3
        coef_default = unique[0]
        unique2 = unique[1:]
        for unq2 in unique2:
            exp_CNF, exp_DNF, exp_CNF3 = self.for_1_filter(unq2, resici, dontcaresici, save=save)
            if save:
                self.save_cnf_dnf(unq2, exp_CNF3, exp_DNF, exp_CNF)
        return exp_CNF, exp_DNF, exp_CNF3

    def for_1_filter(self, unq2, resici, dontcaresici, save=True):
        self.sousblockici = 0
        answer = resici == unq2
        dfres = pd.DataFrame(answer)
        dc = dontcaresici == 1
        dfres.columns = ["Filter_" + str(self.filterici) + "_Value_" + str(int(unq2))]
        dfdontcare = pd.DataFrame(dc)
        dfdontcare.columns = [
            "Filter_" + str(self.filterici) + "_dontcares_" + str(int(unq2))
        ]
        df2 = pd.concat([self.df, dfres, dfdontcare], axis=1)
        if save and self.path_save_exp is not None:
            df2.to_csv(
                self.path_save_exp
                + "Truth_Table_block"
                + str(self.blockici)
                + "_filter_"
                + str(self.filterici)
                + "_coefdefault_"
                + str(unq2)
                + "_sousblock_"
                + str(self.sousblockici)
                + ".csv"
            )
        condtion_filter = df2["index"].values[answer].tolist()
        dc_filter = df2["index"].values[dc].tolist()
        exp_DNF, exp_CNF = self.get_expresion_methode1(
            condtion_filter, dc_filter=dc_filter
        )
        exp_CNF3 = get_exp_with_y(exp_DNF, exp_CNF)
        return exp_CNF, exp_DNF, exp_CNF3

    def get_expresion_methode1(self, condtion_filter, dc_filter=None):
        global dc_filtervf
        self.with_contradiction = True
        self.dontcares_train = []
        if dc_filter is not None:
            dc_filtervf = dc_filter + self.dontcares_train
            condtion_filter_vf = [x for x in condtion_filter if x not in dc_filtervf]
        else:
            condtion_filter_vf = condtion_filter
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
        varici = [
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
            varici[: self.n], minterms=condtion_filter_vf, dontcares=dc_filtervf
        )
        if self.with_contradiction:
            exp_CNF = POSform(
                varici[: self.n], minterms=condtion_filter_vf, dontcares=dc_filtervf
            )
        else:
            exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
        return exp_DNF, exp_CNF


class LTT_AE_1D(LTT_LR_1D):
    def __init__(
        self,
        chanel_interest=1,
        kernel_size=8,
        device="cpu",
        c_a_ajouter=0,
        T=0,
        filter_size=32,
        padding=0,
        stride=5,
        ampli=32,
        kernel_size2=1,
        dropoutLTT=0,
    ):
        super(LTT_AE_1D, self).__init__()
        self.out_float = None
        self.chanel_interest = chanel_interest
        self.k = kernel_size
        self.k2 = kernel_size2
        self.n = self.chanel_interest * self.k
        self.device = device
        self.c_a_ajouter = c_a_ajouter
        self.T = T
        self.s = stride
        self.p = padding
        self.filter_size = filter_size
        self.conv1 = nn.Conv1d(
            1, ampli * filter_size, kernel_size=self.k, stride=self.s, padding=self.p
        )
        self.BN_AE11 = nn.BatchNorm1d(ampli * filter_size)
        self.conv2 = nn.Conv1d(
            ampli * filter_size, filter_size, kernel_size=self.k2, stride=1, padding=0
        )
        self.BN_AE12 = nn.BatchNorm1d(filter_size)
        self.nonlin = act(self.T)
        self.dropoutLTT = nn.Dropout1d(dropoutLTT)

    def forward(self, X):
        self.emmbed = F.gelu(self.BN_AE11(self.dropoutLTT(self.conv1(X.float()))))
        self.out_float = self.BN_AE12(self.conv2(self.emmbed.float()))
        out = self.nonlin(self.out_float)
        return out
