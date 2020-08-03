import split_folders
import os
import numpy as np
import torch
import pickle


def split_into_tvt(inPath, proportions, outPath=None):
    """ proportions should be a list [a, b, c] where a+b+c = 1 """
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


def pickle_datasets(path):
    """path is a directory containing train, val, and test folders"""
    # get data and labels as tensors
    trainData, trainLabels = produce_datasets(path + "train/")
    valData, valLabels = produce_datasets(path + "val/")
    testData, testLabels = produce_datasets(path + "test/")

    trainSet = torch.utils.data.TensorDataset(trainData, trainLabels)
    valSet = torch.utils.data.TensorDataset(valData, valLabels)
    testSet = torch.utils.data.TensorDataset(testData, testLabels)

    pickle.dump(trainSet, open("trainSet1.pkl", "ab"))
    pickle.dump(valSet, open("valSet1.pkl", "ab"))
    pickle.dump(testSet, open("testSet1.pkl", "ab"))

    print("Pickled Data")


def load_data(pathTrain, pathVal, pathTest, batchSize):

    # load pickles
    trainSet = pickle.load(open(pathTrain, "rb"))
    valSet = pickle.load(open(pathVal, "rb"))
    testSet = pickle.load(open(pathTest, "rb"))
    # loader generation
    params = {"batch_size": batchSize, "shuffle": True, "num_workers": 1}

    train_loader = torch.utils.data.DataLoader(trainSet, **params)
    val_loader = torch.utils.data.DataLoader(valSet, **params)
    test_loader = torch.utils.data.DataLoader(testSet, **params)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    pass
