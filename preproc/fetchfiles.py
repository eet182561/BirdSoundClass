#fetch file
'''
This file depends on file_operation. file_operation's read_csv_file program
This function should be merged into a single file later.
'''
import keras
import random
import librosa
import numpy as np
import math

from scipy import signal
from scipy.fft import fftshift

class BirdSequence(keras.utils.Sequence):
    def __init__(self,filenames,labels,batch_size=32,duration=10,sampling_rate=256,shuffle = True):
        '''
        filenames : list of path of filenames to the .mp3. Please note that separate the file names for train and test before calling this function.
        labels: the list of labels
        batch_size : scalar value typically 32 or 64 or power of 2
        '''
        self.x = filenames
        self.y = labels
        self.batch_size = batch_size
        if shuffle:
            temp = list(zip(self.x,self.y))
            random.shuffle(temp)
            self.x,self.y = zip(*temp)
        self.duration = duration #in seconds
    
    def __getitem__(self,idx):
        batch_x_file_names = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_x = []
        for name in batch_x_file_names:
            clip, sample_rate = librosa.load(name, sr=None, duration=self.duration)
            batch_x.append(clip)
        return np.array(batch_x), np.array(batch_y)
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

class BirdSTFT(keras.utils.Sequence):
    def __init__(self,filenames,labels,batch_size=32,duration=10,resampling_rate=40000,shuffle = True):
        '''
        filenames : list of path of filenames to the .mp3. Please note that separate the file names for train and test before calling this function.
        labels: the list of labels
        batch_size : scalar value typically 32 or 64 or power of 2
        '''
        self.x = filenames
        self.y = labels
        self.batch_size = batch_size
        if shuffle:
            temp = list(zip(self.x,self.y))
            random.shuffle(temp)
            self.x,self.y = zip(*temp)
        self.duration = duration #in seconds
        self.resampling_rate = resampling_rate
    
    def __getitem__(self,idx):
        batch_x_file_names = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_x = []
        for name in batch_x_file_names:
            clip, sample_rate = librosa.load(name, sr=None, duration=self.duration)
            clip_res = librosa.resample(clip, sample_rate, self.resampling_rate)
            S = np.abs(librosa.core.stft(clip_res))
            batch_x.append(S)
        return batch_x, np.array(batch_y)
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
