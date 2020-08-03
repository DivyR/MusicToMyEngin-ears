# Relevant imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from SplitTVT import load_data


# CRNN Architecture
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.name = "CRNN"
        self.conv1 = nn.Conv2d(1, 64, 3)  # channels in, channels out, kernel
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.pool1 = nn.MaxPool2d((2, 2), 1)
        self.pool2 = nn.MaxPool2d((4, 2), 1)
        self.GRU1 = nn.GRU(32, 20, dropout=0.3)
        # self.GRU2 = nn.GRU(32, 20, dropout = 0.3)
        self.linear = nn.Linear(20, 6)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.pool2(F.elu(self.conv3(x)))
        x = self.pool2(F.elu(self.conv4(x)))
        x = x.reshape(32, 256, 201)
        print(x.shape)
        x, states = self.GRU1(x)
        # x = self.GRU2(x)
        x = F.softmax(self.linear(x))
        return x


def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.
    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt

    train_err = np.loadtxt("{}_train_acc.csv".format(path))
    val_err = np.loadtxt("{}_val_acc.csv".format(path))
    plt.title("Train vs Validation Accuracy")
    n = len(train_err)  # number of epochs
    plt.plot(range(1, n + 1), train_err, label="Train")
    plt.plot(range(1, n + 1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()


def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values
    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(
        name, batch_size, learning_rate, epoch
    )
    return path


def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    model = model.float()
    for labels, imgs in data_loader:
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        output = model(imgs.float())
        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


def train(model, train_loader, val_loader, batch_size, learn_rate, num_epochs=30):

    torch.manual_seed(1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    train_acc, val_acc = [], []
    train_loss, val_loss = [], []
    # training
    print("Training Started...")
    n = 0  # the number of iterations
    model = model.float()
    adder = 0
    for epoch in range(num_epochs):
        for labels, imgs in iter(train_loader):
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            out = model(imgs.float())  # forward pass
            labels = labels.squeeze(1)
            loss = criterion(out, labels)  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch
            n += 1

        # track accuracy
        train_loss += [loss.item()]
        train_acc.append(get_accuracy(model, train_loader))
        val_acc.append(get_accuracy(model, val_loader) + adder)
        print(epoch, train_acc[-1], val_acc[-1], train_loss[-1])
        model_path = get_model_name(model.name, batch_size, learn_rate, epoch)
        if val_acc[-1] == max(val_acc):  # only save in this case!!
            torch.save(model.state_dict(), model_path)
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    return train_acc, val_acc


use_cuda = True
trainload, valload, testload = load_data(
    "Data-G/trainSet.pkl", "Data-G/valSet.pkl", "Data-G/testSet.pkl", 32
)
net = CRNN()
train(net, trainload, valload, 32, 0.0001)
model_path = get_model_name("net", batch_size=32, learning_rate=0.0001, epoch=29)
plot_training_curve(model_path)
