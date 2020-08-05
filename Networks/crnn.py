import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, name, hiddenSize=10, numLayers=1):
        super(CRNN, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 2, 5)  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(2, 3, 8)
        self.avgpool = nn.AvgPool2d(2, 2)  # kernal size, stride
        self.rnn = nn.GRU(
            input_size=3,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hiddenSize, 6)

    def forward(self, x):
        # CONVOLUTION
        x = x.unsqueeze(1)  # [batch, 20, 213]  -> [batch, 1, 20, 213]
        x = F.relu(self.conv1(x))  # [batch, 1, 20, 213] -> [batch, 10, 16, 209]
        x = self.avgpool(x)  # [batch, 10, 16, 209] -> [batch, 10, 8, 104]
        x = F.relu(self.conv2(x))  # [batch, 20, 8, 104] -> [batch, 20, 1, 97]

        # RNN
        x = x.squeeze(2)  # [batch, 20, 1, 97] -> [batch, 20, 97]
        # each channel is a feature where the x-axis represents time,
        # so we want sequences in the x-direction each with 1 feature from each channel
        x = x.transpose(1, 2)  # [batch, 97, 20]
        out, _ = self.rnn(x)

        # CLASSIFIER
        x = torch.max(out, dim=1)[0]  # [batch, hiddenSize]
        x = self.fc1(x)  # [batch, hiddenSize] -> [batch, 6=genres]
        return x


class CRNN1(nn.Module):
    def __init__(self, name, hiddenSize=10, numLayers=1):
        super(CRNN1, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 25, [3, 2])  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(25, 50, [4, 3])
        self.conv3 = nn.Conv2d(50, 100, 3)
        self.maxpool = nn.MaxPool2d(2, 2)  # kernal size, stride
        self.rnn = nn.GRU(
            input_size=100,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hiddenSize, 6)

    def forward(self, x):
        # CONVOLUTION
        x = x.unsqueeze(1)  # [batch, 1, 20, 213]
        x = F.relu(self.conv1(x))  # [batch, 25, 18, 212]
        x = self.maxpool(x)  # [batch, 25, 9, 106]
        x = F.relu(self.conv2(x))  # [batch, 50, 6, 104]
        x = self.maxpool(x)  # [batch, 50, 3, 52]
        x = F.relu(self.conv3(x))  # [batch, 100, 1, 49]

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


class CRNN2(nn.Module):
    def __init__(self, name, hiddenSize=10, numLayers=1):
        super(CRNN2, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 25, [3, 2])  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(25, 50, [4, 3])
        self.conv3 = nn.Conv2d(50, 100, 3)
        self.maxpool = nn.MaxPool2d(2, 2)  # kernal size, stride
        self.rnn = nn.GRU(
            input_size=100,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hiddenSize, 6)

    def forward(self, x):
        # CONVOLUTION
        x = x.unsqueeze(1)  # [batch, 1, 20, 213]
        x = F.relu(self.conv1(x))  # [batch, 25, 18, 212]
        x = self.maxpool(x)  # [batch, 25, 9, 106]
        x = F.relu(self.conv2(x))  # [batch, 50, 6, 104]
        x = self.maxpool(x)  # [batch, 50, 3, 52]
        x = F.relu(self.conv3(x))  # [batch, 100, 1, 49]

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


class CRNN3(nn.Module):
    def __init__(self, name, hiddenSize=70, numLayers=2):
        super(CRNN3, self).__init__()
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
            dropout=0.5,
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

