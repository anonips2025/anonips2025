import numpy as np
import torch
from icecream import ic
from torch import nn

from src.ttnet.classifier import Classifier, ClassifierBNN
from src.ttnet.LTT_block import LTT_AE_1D, LTT_LR_1D
from src.ttnet.modules import Binarize01Act
from src.ttnet.preprocess import Preprocess

act = Binarize01Act


class TTnet_general(nn.Module):
    def __init__(
        self,
        features_size=100,
        index=94,
        oneforall=True,
        T=0.0,
        chanel_interest=1,
        k=4,  # kernel_size
        device="cpu",
        c_a_ajouter=0,
        filter_size=16,
        p=0,  # padding
        s=4,  # stride
        embed_size=50,
        repeat=3,
        block_LTT="LR",
        regression=False,
        dropoutclass=0.0,
        dropoutLTT=0.0,
        nclass=2,
        poly_flag=False,
        classifier="linear",
    ):
        super(TTnet_general, self).__init__()

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
        self.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
        with torch.no_grad():
            x_r = torch.zeros((1, features_size))
            ic(x_r.shape)
            # print(x_r.shape)
            x_r = self.preprocess0(x_r)
            ic(x_r.shape)
            # print(x_r.shape)
            x_r2 = self.block(x_r)
            ic(x_r2.shape)
            x_r2 = x_r2.reshape(x_r2.shape[0], -1)
            ic(x_r2.shape)
            x_r2 = x_r2.reshape(x_r2.shape[0], -1)
        self.train()  # Set back to training mode
        features_size_LR = x_r2.shape[-1]
        classifier_types = ("linear", "binary")
        if classifier == "linear":
            self.classifier = Classifier(
                embed_size=embed_size,
                features_size_LR=features_size_LR,
                kernel_size=k,
                regression=regression,
                dropoutclass=dropoutclass,
                nclass=nclass,
                poly_flag=poly_flag,
            )
        elif classifier == "binary":
            self.classifier = ClassifierBNN(
                embed_size=embed_size,
                features_size_LR=features_size_LR,
                kernel_size=k,
                regression=regression,
                dropoutclass=dropoutclass,
                nclass=nclass,
                poly_flag=poly_flag,
            )
        else:
            raise f"Not a classifier: {classifier}\nShould be in {classifier_types}"

        print("Model")
        print("Preprocess", self.preprocess0)
        print("Block", self.block)
        print("Classifier", self.classifier)
        self.float()

    def forward(self, X):
        X = self.preprocess0(X.float())

        X = self.block(X)
        X = self.classifier(X)
        return X.float()

    def convert(self):
        truth_table = self.block.get_TT_block_all_filter()
        if len(truth_table.shape) == 1:
            truth_table = truth_table.reshape((truth_table.shape[-1], 1))
        return truth_table
