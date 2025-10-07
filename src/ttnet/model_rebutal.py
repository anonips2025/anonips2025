import numpy as np
import torch
from torch import nn

from src.ttnet.classifier import Classifier
from src.ttnet.LTT_block import LTT_AE_1D, LTT_LR_1D
from src.ttnet.modules import Binarize01Act
from src.ttnet.preprocess import Preprocess

act = Binarize01Act


class TTnet_local(nn.Module):
    def __init__(
        self,
        features_size=100,
        index=94,
        oneforall=True,
        T=0.0,
        chanel_interest=16,
        k=5,
        device="cpu",
        c_a_ajouter=0,
        filter_size=10,
        p=0,
        s=5,
        embed_size=10,
        repeat=3,
        block_LTT="LR",
        regression=False,
        dropoutclass=0.0,
        dropoutLTT=0.0,
        nclass=2,
        poly_flag=False,
    ):
        super(TTnet_local, self).__init__()

        self.kernel = k

        self.preprocess0 = Preprocess(
            index=index,
            features_size=features_size,
            oneforall=oneforall,
            T=T,
            reapeat_c=repeat,
        )
        if block_LTT == "LR":
            self.block = LTT_LR_1D(
                chanel_interest=chanel_interest,
                kernel_size=k,
                device=device,
                c_a_ajouter=c_a_ajouter,
                T=T,
                filter_size=filter_size,
                padding=p,
                stride=s,
                dropoutLTT=dropoutLTT,
            )
        elif block_LTT == "AE":
            self.block = LTT_AE_1D(
                chanel_interest=chanel_interest,
                kernel_size=k,
                device=device,
                c_a_ajouter=c_a_ajouter,
                T=T,
                filter_size=filter_size,
                padding=p,
                stride=s,
                dropoutLTT=dropoutLTT,
            )

        else:
            raise "PB name AE"
        # print(features_size)
        with torch.no_grad():
            x_r = torch.zeros((13, features_size))
            # print(x_r.shape)
            xr1 = self.preprocess0(x_r)
            xr1 = xr1.transpose(0, 1)
            # print(xr1.shape)
            x_r2 = self.block(xr1)
            x_r2 = x_r2.reshape(x_r2.shape[0], -1)
            x_r2 = x_r2.reshape(x_r2.shape[0], -1)
        features_size_LR = filter_size  # x_r2.shape[-1]
        self.classifier = Classifier(
            embed_size=embed_size,
            features_size_LR=features_size_LR,
            kernel_size=k,
            regression=regression,
            dropoutclass=dropoutclass,
            nclass=nclass,
            poly_flag=poly_flag,
        )

        print("Model")
        print("Preprocess", self.preprocess0)
        print("Block", self.block)
        print("Classifier", self.classifier)
        self.float()

    def forward(self, X):
        X = self.preprocess0(X.float())
        # print(X.shape)
        X = nn.ZeroPad2d((0, 5))(X)
        # print(X.shape)
        X = X.reshape(X.shape[0], -1, self.kernel)
        # print(X.shape)
        # ok
        X = self.block(X)
        X = self.classifier(X)
        return X.float()

    def convert(self, pathmodel):
        truth_table = self.block.get_TT_block_all_filter()
        np.save(pathmodel + "/truth_table.npy", truth_table)
        del truth_table
        truth_table = np.load(pathmodel + "/truth_table.npy")
        return truth_table
