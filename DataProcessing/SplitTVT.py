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
        for file in os.listdir(path + folder)
        for folder in os.listdir(path)
    ]


def load_data(path, batchSize):
    """path is a directory containing train, val, and test folders"""
    trainSet = produce_datasets(path + "train/")


if __name__ == "__main__":
    path = "./Data/Dataset/"
    load_data(path, 32)
