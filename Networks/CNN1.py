import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from SplitTVT import load_data
from torchsummary import summary

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
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.name = "net"
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.fc1 = nn.Linear(1530, 50)
        self.fc2 = nn.Linear(50, 6)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1530)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

def train(model, train_loader, val_loader, batch_size, learn_rate, num_epochs=30):

    torch.manual_seed(1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    train_acc, val_acc = [], []
    train_loss, val_loss = [], []
    # training
    print ("Training Started...")
    n = 0 # the number of iterations
    model = model.float()
    adder = 0
    for epoch in range(num_epochs):
        for labels, imgs in iter(train_loader):
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            out = model(imgs.float())             # forward pass
            labels = labels.squeeze(1)
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            n += 1

        # track accuracy
        train_loss += [loss.item()]
        train_acc.append(get_accuracy(model, train_loader))
        val_acc.append(get_accuracy(model, val_loader)+adder)
        print(epoch, train_acc[-1], val_acc[-1], train_loss[-1])
        model_path = get_model_name(model.name, batch_size, learn_rate, epoch)
        if val_acc[-1] == max(val_acc): # only save in this case!!
            torch.save(model.state_dict(), model_path)
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    return train_acc, val_acc


def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    model = model.float()
    for labels, imgs in data_loader:
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        output = model(imgs.float())
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


use_cuda = True
trainload, valload, testload = load_data("Data2/trainSet.pkl", "Data2/valSet.pkl", "Data2/testSet.pkl", 32)
net11 = Network()
train(net11, trainload, valload, 32, 0.0001)
model_path = get_model_name("net", batch_size=32, learning_rate=0.0001, epoch=29)
plot_training_curve(model_path)
