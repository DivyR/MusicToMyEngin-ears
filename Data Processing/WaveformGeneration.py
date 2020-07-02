from pydub import AudioSegment
import os

arr = os.listdir("fma_small")
reject = []
for i in range(0,len(arr)):
    arr2 = os.listdir("fma_small/"+arr[i])
    for j in range(0,len(arr2)):
        path = "fma_small/"+arr[i]+"/"+arr2[j]
        if os.path.getsize(path) > 100000:
            song = AudioSegment.from_mp3("fma_small/"+arr[i]+"/"+arr2[j])
            song = song.set_channels(1)
            name = os.path.splitext(arr2[j])[0]
            song.export("newFMA/"+arr[i]+"/"+name+".wav", format="wav")
        else:
            reject += [arr2[j]]
print(reject)
