import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Convolution->RNN->Classifier"""

    def __init__(self):

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 6))

    def forward(self, X):
        # inputs are: [batch_size, 1, 13, 1250]
        self.conv1(X)  # ouputs are: [batch_size, 5, 11, 1245]

