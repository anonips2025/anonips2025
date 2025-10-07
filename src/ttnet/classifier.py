import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# import brevitas.nn as qnn
from src.ttnet.modules import (
    Binarize01Act,
    BinLinearPos,
    Polynome_ACT,
    TernaryWeightWithMaskFn,
    quantization_int,
)

act = Binarize01Act


g_weight_binarizer = TernaryWeightWithMaskFn


class Classifier(nn.Module):
    def __init__(
        self,
        embed_size=10,
        features_size_LR=100,
        kernel_size=8,
        regression=False,
        dropoutclass=0.0,
        nclass=2,
        poly_flag=False,
    ):
        super(Classifier, self).__init__()

        self.poly_flag = poly_flag
        self.one_layer_flag = False
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.features_size_LR = features_size_LR
        print("Number of rules, ", features_size_LR, " size rules ", self.kernel_size)
        self.dense1 = nn.Linear(features_size_LR, embed_size, bias=False)
        self.BN1 = nn.BatchNorm1d(embed_size)
        self.regression = regression
        self.nclass = nclass
        if self.poly_flag:
            self.Polynome_ACT = Polynome_ACT()
        self.drop = nn.Dropout1d(dropoutclass)
        if self.one_layer_flag:
            assert embed_size == 2
            assert self.poly_flag == False
        else:
            if regression or nclass == 2:
                self.output = nn.Linear(embed_size, 1, bias=False)
            else:
                self.output = nn.Linear(embed_size, nclass, bias=False)

    def forward(self, X, **kwargs):
        X = X.reshape(X.shape[0], -1)
        X = self.drop(self.dense1(X))
        X = self.BN1(X)
        if self.one_layer_flag:
            X = F.softmax(X, dim=-1)
        else:
            if self.poly_flag:
                X = self.Polynome_ACT(X)
            X = self.output(X)
            if self.regression:
                # No additional activation function for regression
                pass
            elif self.nclass == 2:
                X = torch.sigmoid(
                    X
                )  # Sigmoid activation function for binary classification
            else:
                X = F.softmax(
                    X, dim=-1
                )  # Softmax activation function for binary classification
        return X.float()

    def save_lr(self, path_model, coef=10, nbit1=0, nbit2=0):
        var_BN = self.BN1.running_var
        mean_BN = self.BN1.running_mean.cpu()
        eps_BN = self.BN1.eps  # .item()
        gama_BN = self.BN1.weight  # .item()
        beta_BN = self.BN1.bias.cpu()  # .item()
        std_BN = torch.sqrt(var_BN + eps_BN)
        scale = (gama_BN / std_BN).data.cpu().clone().detach().cpu().numpy()
        bias = (beta_BN - mean_BN * scale).data
        W = coef * self.dense1.weight.data.cpu().clone().detach().cpu().numpy()
        if nbit1 != 0:
            W = quantization_int(W, nbit1)
        W = W * np.expand_dims(scale, axis=1)
        b = coef * 1.0 * bias
        if nbit1 != 0:
            b = quantization_int(b, nbit1)
        if self.one_layer_flag:
            pass
        else:
            if self.poly_flag:
                raise "PB not implemented yet"
            W_out = coef * self.output.weight.data.cpu().clone().detach().numpy()
            if nbit2 != 0:
                W_out = quantization_int(W_out, nbit2)
            W_vf = (np.dot(W_out, W)).astype(int)
            B_vf = (np.dot(W_out, b)).astype(int)
        np.save(
            path_model
            + "/W_vf_"
            + str(coef)
            + "_"
            + str(nbit1)
            + "_"
            + str(nbit2)
            + ".npy",
            W_vf,
        )
        np.save(
            path_model
            + "/B_vf_"
            + str(coef)
            + "_"
            + str(nbit1)
            + "_"
            + str(nbit2)
            + ".npy",
            B_vf,
        )
        del W_vf, B_vf
        W_vf = np.load(
            path_model
            + "/W_vf_"
            + str(coef)
            + "_"
            + str(nbit1)
            + "_"
            + str(nbit2)
            + ".npy"
        )
        B_vf = np.load(
            path_model
            + "/B_vf_"
            + str(coef)
            + "_"
            + str(nbit1)
            + "_"
            + str(nbit2)
            + ".npy"
        )
        return W_vf, B_vf


class ClassifierBNN(nn.Module):
    def __init__(
        self,
        embed_size=10,
        features_size_LR=100,
        kernel_size=8,
        regression=False,
        dropoutclass=0.0,
        nclass=2,
        poly_flag=False,
    ):
        super(ClassifierBNN, self).__init__()

        self.poly_flag = poly_flag
        self.one_layer_flag = False
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.features_size_LR = features_size_LR
        print("Number of rules, ", features_size_LR, " size rules ", self.kernel_size)
        self.dense1 = BinLinearPos(g_weight_binarizer, features_size_LR, embed_size)
        self.BN1 = nn.BatchNorm1d(embed_size)
        self.regression = regression
        if self.poly_flag:
            self.Polynome_ACT = Polynome_ACT()
        self.drop = nn.Dropout1d(dropoutclass)
        if self.one_layer_flag:
            assert embed_size == 2
            assert self.poly_flag == False
        else:
            if regression:
                self.output = BinLinearPos(g_weight_binarizer, embed_size, 1)
            else:
                self.output = BinLinearPos(g_weight_binarizer, embed_size, nclass)

    def forward(self, X, **kwargs):
        X = X.reshape(X.shape[0], -1)
        X = self.drop(self.dense1(X))
        X = self.BN1(X)
        if self.one_layer_flag:
            X = F.softmax(X, dim=-1)
        else:
            if self.poly_flag:
                X = self.Polynome_ACT(X)

            if self.regression:
                X = self.output(X)
                # print(X)
            else:
                X = F.softmax(self.output(X), dim=-1)
        return X.float()

    def save_lr(self, path_model, coef=10, nbit1=0, nbit2=0):
        var_BN = self.BN1.running_var
        mean_BN = self.BN1.running_mean
        eps_BN = self.BN1.eps  # .item()
        gama_BN = self.BN1.weight  # .item()
        beta_BN = self.BN1.bias  # .item()
        std_BN = torch.sqrt(var_BN + eps_BN)
        scale = (gama_BN / std_BN).data.cpu().clone().detach().numpy()
        bias = (beta_BN - mean_BN * scale).data.cpu().clone().detach().numpy()
        W = coef * self.dense1.weight.data.cpu().clone().detach().numpy()
        if nbit1 != 0:
            W = quantization_int(W, nbit1)
        W = W * np.expand_dims(scale, axis=1)
        b = coef * 1.0 * bias
        if nbit1 != 0:
            b = quantization_int(b, nbit1)
        if self.one_layer_flag:
            pass
        else:
            if self.poly_flag:
                raise "PB not implemented yet"
            W_out = coef * self.output.weight.data.cpu().clone().detach().numpy()
            if nbit2 != 0:
                W_out = quantization_int(W_out, nbit2)
            W_vf = (np.dot(W_out, W)).astype(int)
            B_vf = (np.dot(W_out, b)).astype(int)
        np.save(
            path_model
            + "/W_vf_"
            + str(coef)
            + "_"
            + str(nbit1)
            + "_"
            + str(nbit2)
            + ".npy",
            W_vf,
        )
        np.save(
            path_model
            + "/B_vf_"
            + str(coef)
            + "_"
            + str(nbit1)
            + "_"
            + str(nbit2)
            + ".npy",
            B_vf,
        )
        del W_vf, B_vf
        W_vf = np.load(
            path_model
            + "/W_vf_"
            + str(coef)
            + "_"
            + str(nbit1)
            + "_"
            + str(nbit2)
            + ".npy"
        )
        B_vf = np.load(
            path_model
            + "/B_vf_"
            + str(coef)
            + "_"
            + str(nbit1)
            + "_"
            + str(nbit2)
            + ".npy"
        )
        return W_vf, B_vf
