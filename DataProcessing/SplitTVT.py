import split_folders
import os
import numpy as np
import torch


def split_into_tvt(inPath, outPath=None, proportions=(0.7, 0.15, 0.15)):
    if outPath is None:
        outPath = inPath
    split_folders.ratio(input=inPath, output=outPath, seed=22, ratio=proportions)
    return


def produce_datasets(path):
    """Assumes that the path is a train, val, or test folder"""
    # import all images as numpy array into a list
    data, labels = [], []
    for folder in os.listdir(path):
        data += [
            np.loadtxt("{}{}/{}".format(path, folder, file))[:, :1292]
            for file in os.listdir(path + folder)
        ]

        label = int(float(folder))
        labels += [np.array([label]) for _ in range(len(os.listdir(path + folder)))]

    return torch.tensor(labels), torch.tensor(data)


def load_data(path, batchSize):
    """path is a directory containing train, val, and test folders"""

    # get data and labels as tensors
    print("1")
    trainData, trainLabels = produce_datasets(path + "train/")
    print("2")
    valData, valLabels = produce_datasets(path + "val/")
    print("3")
    testData, testLabels = produce_datasets(path + "test/")

    print("4")
    trainSet = torch.utils.data.TensorDataset(trainData, trainLabels)
    valSet = torch.utils.data.TensorDataset(valData, valLabels)
    testSet = torch.utils.data.TensorDataset(testData, testLabels)

    # loader generation
    params = {"batch_size": batchSize, "shuffle": True, "num_workers": 1}

    print("5")
    train_loader = torch.utils.data.DataLoader(trainSet, **params)
    print("6")
    val_loader = torch.utils.data.DataLoader(valSet, **params)
    test_loader = torch.utils.data.DataLoader(testSet, **params)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    path = "./Data/Dataset/"
    train_loader, _, _ = load_data(path, 100)
