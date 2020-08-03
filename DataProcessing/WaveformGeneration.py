from scipy.io.wavfile import read
from scipy import signal
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
import numpy as np
import librosa


def split_file(path, filename, exportpath):
    song = AudioSegment.from_wav(path+filename)
    name = os.path.splitext(filename)[0]
    genre = os.path.splitext(name)[0]
    for i in range(0, 6):
        split = song[i*5000:(i+1)*5000]
        split.export(exportpath+name+"-"+str(i)+".wav", format="wav")



def split_dataset():
    arr = sorted(os.listdir("genres"))
    for i in range(0,len(arr)):
        arr2 = sorted(os.listdir("genres/"+arr[i]))
        for j in range(0,len(arr2)):
            path = "genres/"+arr[i]+"/"
            filename = arr2[j]
            exportpath = "NewDataSet/"+arr[i]+"/"
            split(path,filename,exportpath)




def mfcc(foldername,exportfolder):
    arr = sorted(os.listdir(foldername))
    for i in range(0,len(arr)):
        arr2 = sorted(os.listdir(foldername+"/"+arr[i]))
        for j in range(0,len(arr2)):
            path = foldername+"/"+arr[i]+"/"+arr2[j]
            y, s2 = librosa.load(path)
            S = librosa.feature.melspectrogram(y, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
            name = os.path.splitext(arr2[j])[0]
            np.savetxt(exportfolder+"/"+arr[i]+"/"+name+".txt", mfcc)
