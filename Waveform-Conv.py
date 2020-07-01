from scipy.io.wavfile import read
from scipy import signal
import matplotlib.pyplot as plt
from pydub import AudioSegment
input_data = read("file2.wav")
audio = input_data[1]
# plot the first 1024 samples
plt.plot(audio[0:10000])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title
plt.title("Sample Wav")
# display the plot
plt.show()

#song = AudioSegment.from_mp3("fma_small/000/000002.mp3")
#song = song.set_channels(1)
#song.export("file2.wav", format="wav")

#sample_rate, samples = read('file.wav')
#frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

#plt.pcolormesh(times, frequencies, spectrogram)
#plt.imshow(spectrogram)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
