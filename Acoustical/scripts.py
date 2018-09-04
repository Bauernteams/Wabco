
# Record Audio Data
import pyaudio
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import time
import wave
import os
import sys

        
#######################        TestID & Unterordner erstellen         #######################
TEST_ID = input("Bitte TestID eingeben: ")
#folder = str(testID) + "/" + str(testIDRound) 
folder = "Q:/Repositories/Wabco/Datastore/Acoustical/180723-24/Phillips/"

CHUNK = 16384

WIDTH = 2
DTYPE = np.int16
MAX_INT = 32768.0
FORMAT = pyaudio.paInt16

CHANNELS = 1
RATE = 48000
WAVE_OUTPUT_FILENAME = TEST_ID

FILEPATH = os.path.join(folder,WAVE_OUTPUT_FILENAME+".wav")

if os.path.isfile(FILEPATH):
    answer = input("Diese TestID existiert bereits. (o)verwrite or (q)uit?")
    if answer == "q":
        sys.exit(0)
    elif answer == "o":
        print("overwriting", FILEPATH)
    else:
        sys.exit("Entered unknown input:", answer)


p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

print("* recording")
print("...ctrl+C to end recording")
# initialize sample buffer
buffer = np.zeros(CHUNK)

allData = []
try:
    while True:
        string_audio_data = stream.read(CHUNK)
        allData.append(string_audio_data)
except KeyboardInterrupt:
    pass
print("finished recording") 
# stop Recording
stream.stop_stream()
stream.close()
p.terminate()
 
waveFile = wave.open(FILEPATH, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(p.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(allData))
waveFile.close()