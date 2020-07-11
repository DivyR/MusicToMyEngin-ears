import os
import pandas as pd
import numpy as np
import shutil

df = pd.read_pickle("labels.pkl")
arr = sorted(os.listdir("FMA"))
count = 0
for i in range(0, len(arr)):
    arr2 = sorted(os.listdir("FMA/"+arr[i]))
    for j in range(0, len(arr2)):
        path1 = "FMA/"+arr[i]+"/"+arr2[j]
        arr3 = sorted(os.listdir("Dataset"))
        for k in range(0, len(arr3)):
            if str(df.iloc[count]['Parent Genre']) == arr3[k]:
                shutil.copy(path1, "Dataset/"+arr3[k])
        count += 1
