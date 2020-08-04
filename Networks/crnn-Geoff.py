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
    def __init__(self, name, hiddenSize=50, numLayers=1):
        super(CRNN, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 10, 5)  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(10, 20, 8)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.avgpool = nn.AvgPool2d(2, 2)  # kernal size, stride
        self.rnn = nn.GRU(
            input_size=20,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout = 0.5
        )
        self.fc1 = nn.Linear(hiddenSize, 6)

    def forward(self, x):
        # CONVOLUTION
        x = x.unsqueeze(1)  # [batch, 20, 213]  -> [batch, 1, 20, 213]
        x = F.relu(self.conv1(x))  # [batch, 1, 20, 213] -> [batch, 3, 16, 209]
        x = self.avgpool(x)  # [batch, 3, 16, 209] -> [batch, 3, 8, 104]
        x = F.relu(self.conv2_bn(self.conv2(x)))  # [batch, 3, 8, 104] -> [batch, 6, 1, 97]

        # RNN
        x = x.squeeze(2)  # [batch, 6, 1, 97] -> [batch, 6, 97]
        # each channel is a feature where the x-axis represents time,
        # so we want sequences in the x-direction each with 1 feature from each channel
        x = x.transpose(1, 2)  # [batch, 97, 6]
        out, _ = self.rnn(x)

        # CLASSIFIER
        x = torch.max(out, dim=1)[0]  # [batch, hiddenSize]
        x = self.fc1(x)  # [batch, hiddenSize] -> [batch, 6=genres]
        return x


class CRNN1(nn.Module):
    def __init__(self, name, hiddenSize=75, numLayers=1):
        super(CRNN1, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 10, 5)  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(10, 20, 8)
        self.conv2_bn = nn.BatchNorm2d(10)
        self.avgpool = nn.AvgPool2d(2, 2)  # kernal size, stride
        self.conv2_bn2 = nn.BatchNorm2d(20)
        self.rnn = nn.GRU(
            input_size=20,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout = 0.5,
        )
        self.fc1 = nn.Linear(hiddenSize, 6)

    def forward(self, x):
        # CONVOLUTION
        x = x.unsqueeze(1)  # [batch, 20, 213]  -> [batch, 1, 20, 213]
        x = F.relu(self.conv2_bn(self.conv1(x)))  # [batch, 1, 20, 213] -> [batch, 3, 16, 209]
        x = self.avgpool(x)  # [batch, 3, 16, 209] -> [batch, 3, 8, 104]
        x = F.relu(self.conv2(x))  # [batch, 3, 8, 104] -> [batch, 6, 1, 97]

        # RNN
        x = x.squeeze(2)  # [batch, 6, 1, 97] -> [batch, 6, 97]
        # each channel is a feature where the x-axis represents time,
        # so we want sequences in the x-direction each with 1 feature from each channel
        x = x.transpose(1, 2)  # [batch, 97, 6]
        out, _ = self.rnn(x)

        # CLASSIFIER
        x = torch.max(out, dim=1)[0]  # [batch, hiddenSize]
        x = self.fc1(x)  # [batch, hiddenSize] -> [batch, 6=genres]
        return x


class CRNN2(nn.Module):
    def __init__(self, name, hiddenSize=75, numLayers=1):
        super(CRNN2, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 10, 5)  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(10, 20, 8)
        self.conv2_bn2 = nn.BatchNorm2d(20)
        self.avgpool = nn.AvgPool2d(2, 2)  # kernal size, stride
        self.rnn = nn.GRU(
            input_size=20,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout = 0.4
        )
        self.fc1 = nn.Linear(hiddenSize, 6)

    def forward(self, x):
        # CONVOLUTION
        x = x.unsqueeze(1)  # [batch, 20, 213]  -> [batch, 1, 20, 213]
        x = F.relu(self.conv1(x))  # [batch, 1, 20, 213] -> [batch, 3, 16, 209]
        x = self.avgpool(x)  # [batch, 3, 16, 209] -> [batch, 3, 8, 104]
        x = F.relu(self.conv2(x))  # [batch, 3, 8, 104] -> [batch, 6, 1, 97]

        # RNN
        x = x.squeeze(2)  # [batch, 6, 1, 97] -> [batch, 6, 97]
        # each channel is a feature where the x-axis represents time,
        # so we want sequences in the x-direction each with 1 feature from each channel
        x = x.transpose(1, 2)  # [batch, 97, 6]
        out, _ = self.rnn(x)

        # CLASSIFIER
        x = torch.max(out, dim=1)[0]  # [batch, hiddenSize]
        x = self.fc1(x)  # [batch, hiddenSize] -> [batch, 6=genres]
        return x
