# We'll need numpy for some mathematical operations
import numpy as np

# matplotlib for displaying the output
import matplotlib.pyplot as plt

# and IPython.display for audio output
import IPython.display

# Librosa for audio
import librosa
import os
# And the display module for visualization
import librosa.display

arr = os.listdir("fma_small")
reject = []
for i in range(0,len(arr)):
    arr2 = os.listdir("fma_small/"+arr[i])
    for j in range(0,len(arr2)):
        path = "fma_small/"+arr[i]+"/"+arr2[j]
        y, sr = librosa.load(path)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        name = os.path.splitext(arr2[j])[0]
        np.savetxt("FMA/"+arr[i]+"/"+name+".txt", mfcc)
