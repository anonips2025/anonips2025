import numpy as np
import torch
from torch import nn

from src.ttnet.modules import Binarize01Act

act = Binarize01Act


class Preprocess(nn.Module):
    def __init__(self, index=94, features_size=100, oneforall=True, T=0, reapeat_c=3):
        super(Preprocess, self).__init__()

        self.index0 = index
        self.F = features_size
        self.index1 = self.index0 - self.F
        self.index1_abs = abs(self.index1)
        self.reapeat_c = reapeat_c
        self.oneforall = oneforall
        self.T = T
        self.nonlin = act(self.T)
        # print('index0', self.index0)
        if self.index0 > 0:
            if self.oneforall:
                self.BN0 = nn.BatchNorm1d(1)
                self.BN1 = nn.BatchNorm1d(1)
                self.BN2 = nn.BatchNorm1d(1)
            else:
                self.BN0 = nn.BatchNorm1d(self.index1_abs)
                self.BN1 = nn.BatchNorm1d(self.index1_abs)
                self.BN2 = nn.BatchNorm1d(self.index1_abs)

    def forward(self, X, **kwargs):
        if self.index0 > 0:
            global X_c0, X_c1, X_c2
            X_d, X_c = (
                X[:, : self.index1].clone().unsqueeze(1).float(),
                X[:, self.index0 :].clone(),
            )
            for x in range(self.reapeat_c):
                X_c0 = self.nonlin(self.BN0(X_c.unsqueeze(1).float()))
                X_c1 = self.nonlin(self.BN1(X_c.unsqueeze(1).float()))
                X_c2 = self.nonlin(self.BN2(X_c.unsqueeze(1).float()))
            if self.reapeat_c == 1:
                X = torch.cat((X_d, X_c0), axis=2)
            elif self.reapeat_c == 2:
                X = torch.cat((X_d, X_c0, X_c1), axis=2)
            elif self.reapeat_c == 3:
                X = torch.cat((X_d, X_c0, X_c1, X_c2), axis=2)
            else:
                raise "PB"
        else:
            X = X.unsqueeze(1).float()
        return X

    def save_pr(self, path_model):
        cpt = 0
        scales, biass = [], []
        for bn in [self.BN0, self.BN1, self.BN2]:
            var_BN = bn.running_var
            mean_BN = bn.running_mean
            eps_BN = bn.eps
            gama_BN = bn.weight
            beta_BN = bn.bias
            std_BN = torch.sqrt(var_BN + eps_BN)
            scale = (gama_BN / std_BN).data.cpu().clone().detach().numpy()
            bias = (beta_BN - mean_BN * scale).data.cpu().clone().detach().numpy()
            np.save(path_model + "/preprocess_" + str(cpt) + "_BN_scale.npy", scale)
            np.save(path_model + "/preprocess_" + str(cpt) + "_BN_bias.npy", bias)
            del scale, bias
            scale = np.load(path_model + "/preprocess_" + str(cpt) + "_BN_scale.npy")
            bias = np.load(path_model + "/preprocess_" + str(cpt) + "_BN_bias.npy")
            scales.append(scale)
            biass.append(bias)
            cpt += 1
        return scales[0], scales[1], scales[2], biass[0], biass[1], biass[2]
