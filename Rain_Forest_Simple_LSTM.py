# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:58:16 2020

@author: itsyo
"""

import numpy as np # linear algebra
from numpy import savez_compressed

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
from scipy.interpolate import interp1d
import gc
'''
from cuml.linear_model import LogisticRegression
from cuml.neighbors import KNeighborsClassifier
from cuml.svm import SVC
from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
'''
import soundfile as sf
# Librosa Libraries
# import librosa
# import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import scipy
from scipy import signal

from sklearn.metrics import roc_auc_score, label_ranking_average_precision_score
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.layers import Input, Lambda, merge, Dense, Flatten, Dropout, BatchNormalization, LSTM, Minimum, Concatenate, Masking, Dot, Average
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, LambdaCallback

def extract_fft(fn):
    data, samplerate = sf.read(fn)
    data = np.array(data)

    varfft = np.abs( np.fft.fft(data)[:(len(data)//2)] )
    
    return varfft.reshape( (1000,1440) ).mean(axis=1)

def write_to_disk(fn, stft, base_path = ''):
    fn = fn[fn.rindex('\\')+1:fn.rindex('.')]
    fn_stft = fn
    print(os.path.join(base_path,fn_stft))
    savez_compressed(os.path.join(base_path,fn_stft), stft)

def extract_stft(fn,start_time=0,end_time=60, nperseg=1200,noverlap=600, nfft=2048, pad_dur=4.2):
    data, samplerate = sf.read(fn)
    data = np.array(data)
    data = data[int(np.floor(samplerate*start_time)) : int(np.floor(samplerate*end_time))]
    if data.shape[0] < pad_dur*samplerate:
        data = np.concatenate([data, np.zeros(int(samplerate*pad_dur) - np.size(data))])
    else:
        data = data[0 : int(np.floor(samplerate*pad_dur))]
    return np.abs(signal.stft(data, samplerate, nperseg=nperseg,noverlap=noverlap, nfft=nfft)[2])

def evaluate_model(trainX, trainy):
    verbose, epochs, batch_size = 2, 4, 32
    print('printing netwok shape',trainX.shape[1], trainX.shape[2], trainy.shape[1])
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Masking(mask_value=trainX[0,-1,:], input_shape=(n_timesteps,n_features)))
    model.add(LSTM(128,  return_sequences=False))

    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.8, verbose=verbose)
    print(history.history.keys())
    return model


# create custom genrator class for small dataset

class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, audio_filenames, batch_size) :
    self.audio_filenames = audio_filenames
    self.count = len(audio_filenames)
    self.batch_size = batch_size
    
    
  def __len__(self) :
    # print("length by the class function", (np.ceil(len(self.audio_filenames) / float(self.batch_size))).astype(np.int))
    return (np.ceil(len(self.audio_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    # write code for preprocessing the audio file and accumulating the stft
    # print('index for getting element', idx, (idx +1)* self.batch_size, self.count)
    if (idx +1)* self.batch_size < self.count:
        data  = self.audio_filenames[idx*self.batch_size : (idx+1)*self.batch_size]
    else :
        # print("inside else of getting item", idx, self.batch_size, self.count )
        data = self.audio_filenames[idx*self.batch_size : self.count]
        # data.extend(self.audio_filenames[0:self.count - idx*self.batch_size ])
    # print('length of data =', len(data))
    all_data = []
    all_label = []
    for row in data:
        a = extract_stft(os.path.join(base_path_file_read, row[0]+'.flac'), row[3], row[5])
        y = keras.utils.to_categorical(row[1], num_classes=25)
        all_data.append(a)
        all_label.append(y)
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    return all_data, all_label    



traint = pd.read_csv( 'F:/database/kaggle/train_tp_small.csv' )
train_fp = pd.read_csv('F:/database/kaggle/train_fp_small.csv' )
train_fp['species_id'] = 24
all_data = pd.concat([traint, train_fp])
shuffled_data = shuffle(all_data)
shuffled_data = shuffled_data.values
validation_split = 0.8
training_data = shuffled_data[0: int(len(shuffled_data)*validation_split)]
validation_data = shuffled_data[int(len(shuffled_data)*validation_split) : ]    

train_generator = My_Custom_Generator(training_data, 32)
validation_generator = My_Custom_Generator(validation_data, 32)
base_path_file_read = 'F:\\database\\kaggle\\train'
a = extract_stft(os.path.join(base_path_file_read, traint['recording_id'][0]+'.flac'))
n_classes = 25
batch_size = 64
n_timesteps, n_features, n_outputs = a.shape[0], a.shape[1], n_classes
model = Sequential()
model.add(Masking(mask_value=a[-1,:], input_shape=(n_timesteps,n_features)))
model.add(LSTM(32,  return_sequences=False))

model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#%% training the model

model.fit_generator(generator=train_generator,
                   steps_per_epoch = 10,
                   epochs = 5,
                   verbose = 2,
                   validation_data = validation_generator,
                   validation_steps = 2)

#%% testing the trained model
# take one minute data then calculate the stft for the data and predict the class
# predict the calss in the test data
test_dir = 'F:\\database\\kaggle\\test'
pad_dur = 4.2
noverlap=600
increment = int(48000*pad_dur/noverlap) + 1
df = pd.DataFrame(columns=['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12'
                           , 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])
for (dirpath, dirnames, filenames ) in os.walk(test_dir):
    stft_one_file = []
    for file in filenames:
        print(file)
        file_features = extract_stft(os.path.join(dirpath, file), 0, 60, pad_dur=60)
        sag_feature = np.zeros((int(file_features.shape[1]/increment), file_features.shape[0], increment))
        j = 0
        for i in range(increment, file_features.shape[1], increment):
            sag_feature[j,:,:] = file_features[:, i-increment : i]
            j +=1
        res = np.average(model.predict(sag_feature), axis = 0)
        df = df.append({'recording_id': file, 's0': res[0], 's1': res[1], 's2': res[2], 's3': res[3], 's4': res[4]
                        , 's5': res[5], 's6': res[6], 's7': res[7], 's8': res[8], 's9': res[9], 's10': res[10]
                        , 's11': res[11], 's12': res[12], 's13': res[13], 's14': res[14], 's15': res[15]
                        , 's16': res[16], 's17': res[17], 's18': res[18], 's19': res[19], 's20': res[20], 's21': res[21]
                        , 's22': res[22], 's23': res[23]}, ignore_index=True)
        

df.to_excel('F:\\database\\kaggle\\sample_submission.xlsx', index=False)

        
    
