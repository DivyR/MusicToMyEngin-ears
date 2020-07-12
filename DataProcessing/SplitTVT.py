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
        current = [
            np.loadtxt("{}{}/{}".format(path, folder, file))[:, :1250]
            for file in os.listdir(path + folder)
        ]
        current = [
            current[i] for i in range(len(current)) if not current[i].shape[1] < 1250
        ]
        data += current

        label = int(float(folder))
        labels += [np.array([label]) for _ in range(len(current))]

    return torch.tensor(labels), torch.tensor(data)


def load_data(path, batchSize):
    """path is a directory containing train, val, and test folders"""

    # get data and labels as tensors
    trainData, trainLabels = produce_datasets(path + "train/")
    valData, valLabels = produce_datasets(path + "val/")
    testData, testLabels = produce_datasets(path + "test/")

    trainSet = torch.utils.data.TensorDataset(trainData, trainLabels)
    valSet = torch.utils.data.TensorDataset(valData, valLabels)
    testSet = torch.utils.data.TensorDataset(testData, testLabels)

    # loader generation
    params = {"batch_size": batchSize, "shuffle": True, "num_workers": 1}

    train_loader = torch.utils.data.DataLoader(trainSet, **params)
    val_loader = torch.utils.data.DataLoader(valSet, **params)
    test_loader = torch.utils.data.DataLoader(testSet, **params)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    path = "./Data/Dataset/"
    train_loader, _, _ = load_data(path, 100)
