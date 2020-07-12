import split_folders
import os
import numpy as np


def split_into_tvt(inPath, outPath=None, proportions=(0.7, 0.15, 0.15)):
    if outPath is None:
        outPath = inPath
    split_folders.ratio(input=inPath, output=outPath, seed=22, ratio=proportions)
    return


def produce_datasets(path):
    """Assumes that the path is a train, val, or test folder"""
    # import all images as numpy array into a list
    return [
        np.loadtxt("{}{}/{}".format(path, folder, file))
        for folder in os.listdir(path)
        for file in os.listdir(path + folder)
    ]


def load_data(path, batchSize):
    """path is a directory containing train, val, and test folders"""

    # get datasets
    trainSet = produce_datasets(path + "train/")
    valSet = produce_datasets(path + "val/")
    testSet = produce_datasets(path + "test/")

    # loader generation
    params = {"batch_size": batchsize, "shuffle": True, "num_workers": 1}

    train_loader = torch.utils.data.DataLoader(trainSet, **params)
    val_loader = torch.utils.data.DataLoader(valSet, **params)
    test_loader = torch.utils.data.DataLoader(testSet, **params)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    path = "./Data/Dataset/"
    train_loader, _, _ = load_data(path, 100)
