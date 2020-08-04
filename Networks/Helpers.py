import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(22)


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


def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt

    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    plt.title("Train and Validation Accuracy")
    n = len(train_acc)  # number of epochs
    plt.plot(range(1, n + 1), train_acc, label="Train")
    plt.plot(range(1, n + 1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()

    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Loss")
    plt.plot(range(1, n + 1), train_loss, label="Train")
    plt.plot(range(1, n + 1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()


def get_accuracy(model, data, criterion, batch_size=32):
    """ Compute the accuracy of the `model` across a dataset `data`
    I MODIFIED THIS FUNCTION TO ALSO RETURN THE LOSS
    """
    correct, total = 0, 0
    loss = 0.0
    for i, batch in enumerate(data):
        labels, mfccs = batch
        mfccs = mfccs.float()
        labels = labels.squeeze(1)
        output = model(mfccs)

        loss += criterion(output, labels).item()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total, loss / (i + 1)


def train(model, train, valid, learning_rate, batch_size, num_epochs=30, save=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # accuracy tracking
    train_acc, valid_acc = [], []
    train_loss, valid_loss = [], []

    # training network
    print("Training Started...")
    startTime = time.time()

    for epoch in range(num_epochs):
        for labels, mfccs in train:
            optimizer.zero_grad()
            output = model(mfccs.float())
            loss = criterion(output, labels.squeeze(1))
            loss.backward()
            optimizer.step()

        tacc, tloss = get_accuracy(model, train, criterion, batch_size=batch_size)
        vacc, vloss = get_accuracy(model, valid, criterion, batch_size=batch_size)

        train_acc.append(tacc)
        valid_acc.append(vacc)
        train_loss.append(tloss)
        valid_loss.append(vloss)

        print(
            (
                "Epoch {}: Train acc: {}, Train loss: {} | "
                + "Validation acc: {}, Validation loss: {}"
            ).format(
                epoch + 1, train_acc[-1], train_loss[-1], valid_acc[-1], valid_loss[-1]
            )
        )
        if save:
            if valid_loss[-1] == min(valid_loss) or valid_acc[-1] == max(
                valid_acc
            ):  # only save state in this case!!
                modelPath = get_model_name(
                    model.name, batch_size, learning_rate, epoch + 1
                )
                torch.save(model.state_dict(), modelPath)

    modelPath = get_model_name(model.name, batch_size, learning_rate, epoch + 1)
    np.savetxt("{}_train_acc.csv".format(modelPath), train_acc)
    np.savetxt("{}_val_acc.csv".format(modelPath), valid_acc)
    np.savetxt("{}_train_loss.csv".format(modelPath), train_loss)
    np.savetxt("{}_val_loss.csv".format(modelPath), valid_loss)

    endTime = time.time()
    elapsedTime = endTime - startTime
    print("Total time elapsed: {:.2f} seconds".format(elapsedTime))
    print("Finished Training")
