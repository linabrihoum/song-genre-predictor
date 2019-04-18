
# coding: utf-8




# Song genre classifier
# Machine Learning
# Lina Brihoum
#https://nbviewer.jupyter.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb
# Main source: https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8





import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
import wave
import struct
import matplotlib.pyplot as plt
import os
import pathlib
import csv
import config
import joblib
import os
from os import listdir
from os.path import isfile, join
import csv

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings('ignore')





audio_path = 'memories.wav' # Audio
x , sr = librosa.load(audio_path) # Using librosa library to load audio

# Returns an audio time series as a numpy array with a default sampling rate
print(x.shape, sr)
print(type(x), type(sr))





librosa.load(audio_path, sr=44100)
# Sample at 441.KHz, you can also put sr = None to disable resampling





#Display audio
# ipd.Audio(audio_path)





# Plot of the amplitude envelope of the waveform of the audio file
plt.figure(figsize=(20,5)) # Change the first number to get a bigger sample
librosa.display.waveplot(x, sr=sr)





#Spectrogram of the audio file
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

# Vertical axis shows frequencies from 0 to 10kHZ
# Horizontal axis shows time of clip





# Convert frequency axis to a logarithmic
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()





# Writing audio
librosa.output.write_wav('example.wav', x, sr)





# Creating an audio signal at 220Hz as a numpy array
sr = 22050 # Sample rate
T = 5.0 # Seconds
t = np.linspace(0, T, int(T*sr), endpoint=False) # Time variable
x = 0.5*np.sin(2*np.pi*220*t) # Pure sine wave at 220Hz





# Playing the audio sample
# ipd.Audio(x, rate=sr) #Loading the numpy array





# Saving the audio sample to an external wav file
librosa.output.write_wav('example.wav', x, sr)





# Extracting the characteristics (feature extraction)
# Zero Crossing Rate - rate of sign changes from positive to negative

#Loading the signal
x, sr = librosa.load('memories.wav')

#Plotting the signal
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)





# Zooming into the wave file
n0 = 9000
n1 = 9100
plt.figure(figsize=(10, 5))
plt.plot(x[n0:n1])
plt.grid()





#Verifying how many zero crossings there are
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))





# Computing the spectral centroid for each frame in a signal
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
print(spectral_centroids.shape)





# Computing the time variable for visulization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)





# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)





# Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')





# Spectral Rolloff - measuring of the shape of the signal
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')





# Mel-frequency cepstral coefficients - signal of a small set of features
x, fs = librosa.load('memories.wav')
librosa.display.waveplot(x, sr=sr)





mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)
# The result will mean the mfcc computed 20 MFCC over 16211 frames





# Display the Mel-Frequency cepstral coefficients
librosa.display.specshow(mfccs, sr=sr, x_axis='time')





# Scaling over each coefficient dimension that has
# zero mean and unit variance
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))

librosa.display.specshow(mfccs, sr=sr, x_axis='time')





# Chroma features - bins representing distinct smitones of the musical octave

# Loading the file
x, sr = librosa.load('memories.wav')
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')





# Now getting to the fun part :)
# Classifying songs after analyzing the audio signals





# Extracting the spectogram for every audio
# This takes a while :(

# cmap = plt.get_cmap('inferno')
#
# plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
# for g in genres:
#     pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
#     for filename in os.listdir(f'./genres/{g}'):
#         songname = f'./genres/{g}/{filename}'
#         y, sr = librosa.load(songname, mono=True, duration=5)
#         plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
#         plt.axis('off');
#         plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
#         plt.clf()





# Extracting every feature
# Extracting MFCC, Spectral centroid,
# zero crossing rate, chroma frequencies, & spectral roll off
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 20):
    header += f' mfcc{i}'
header += ' label'
header = header.split()





# Writing all this data to a CSV file
# This took like 30 minutes lol

# file = open('data.csv', 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)
# genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
# for g in genres:
#     for filename in os.listdir(f'./genres/{g}'):
#         songname = f'./genres/{g}/{filename}'
#         y, sr = librosa.load(songname, mono=True, duration=30)
#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)
#         to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
#         for e in mfcc:
#             to_append += f' {np.mean(e)}'
#         to_append += f' {g}'
#         file = open('data.csv', 'a', newline='')
#         with file:
#             writer = csv.writer(file)
#             writer.writerow(to_append.split())





# Print out some of the CSV file
data = pd.read_csv('data.csv')
print(data.head())





# Remove the file name (string) so we can transform the floats to features
data.shape
data = data.drop(['filename'], axis=1)





# Data without filename now
print(data.head())





# Encode the labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)





# Scale the feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-2], dtype = float))





# Dividing data into trainging and testing set, finally :D
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X Test Shape: " + str(X_test.shape))
print("X Test Snippet: " + str(X_test))


print(X_train[10])





# Now classifying the songs after training and testing using TensorFlow
import keras
from keras import models
from keras import layers





model = models.Sequential()
model.add(layers.Dense(1024, activation='relu',
                       input_shape=(X_train.shape[1],)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))





# Compile the models
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])





# Fit the models
history = model.fit(X_train,
                    y_train,
                    epochs=35,
                    batch_size=128)





# Testing loss and testing accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)





print(test_acc)
print(test_loss)





# Since our accuracy is low, we will validate our set
x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]





model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)





# Print results
print("Results: " + str(results))





# Predictions on Test Data
predictions = model.predict(X_test)
print(predictions[0].shape)

print("Predictions: " + str(predictions))




print("Sum of predictions: " + str(np.sum(predictions[0])))





print("Predictions Argmax: " + str(np.argmax(predictions[0])))





songname = f'./blues90.wav'
sample_data = []

for offsetMemes in range(0, 90):
    y, sr = librosa.load(songname, mono=True, duration=30, offset=offsetMemes)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    data = f'{np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        data += f' {np.mean(e)}'
    data += f''
    sample_data.append(data.split())
    offsetMemes = offsetMemes + 30
print("Before: " + str(sample_data))
sample_data = np.array(sample_data).reshape(-1,24)
print("After: " + str(sample_data))

print("Shape Memes: " + str(sample_data.shape))

predictions = model.predict(sample_data)

print(predictions)

print("Its " + str(genres[np.argmax(predictions)]))
