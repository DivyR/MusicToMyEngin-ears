import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from SplitTVT import load_data
import Helpers as hp
from numpy import loadtxt

class CRNN(nn.Module):
    def __init__(self, name, hiddenSize=70, numLayers=2):
        super(CRNN, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 25, [3, 2])  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(25, 50, [4, 3])
        self.conv3 = nn.Conv2d(50, 100, 3)
        self.conv2_bn2 = nn.BatchNorm2d(100)
        self.maxpool = nn.MaxPool2d(2, 2)  # kernal size, stride
        self.rnn = nn.GRU(
            input_size=100,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout = 0.5,
        )
        self.fc1 = nn.Linear(hiddenSize, 6)

    def forward(self, x):
        # CONVOLUTION
        x = x.unsqueeze(1)  # [batch, 1, 20, 213]
        x = F.relu(self.conv1(x))  # [batch, 25, 18, 212]
        x = self.maxpool(x)  # [batch, 25, 9, 106]
        x = F.relu(self.conv2(x))  # [batch, 50, 6, 104]
        x = self.maxpool(x)  # [batch, 50, 3, 52]
        x = F.relu(self.conv2_bn2(self.conv3(x)))  # [batch, 100, 1, 49]

        # RNN
        x = x.squeeze(2)
        # each channel is a feature where the x-axis represents time,
        # so we want sequences in the x-direction each with 1 feature from each channel
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)

        # CLASSIFIER
        x = torch.max(out, dim=1)[0]  # [batch, hiddenSize]
        x = self.fc1(x)  # [batch, hiddenSize] -> [batch, 6=genres]
        return x
