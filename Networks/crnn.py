import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, name, hiddenSize=10, numLayers=1):
        super(CRNN, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 3, 5)  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(3, 6, 8)
        self.avgpool = nn.AvgPool2d(2, 2)  # kernal size, stride
        self.rnn = nn.GRU(
            input_size=6,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
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
