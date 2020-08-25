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
import pickle

from scipy import signal
from scipy.fft import fftshift
from util import to_one_hot

class BirdSequence(keras.utils.Sequence):
    def __init__(self,filenames,labels,batch_size=32,duration=10,sampling_rate=256,shuffle = True,small_files= None):
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
    def __init__(self,filenames,labels,batch_size=32,duration=10,resampling_rate=48000,shuffle = True,small_files = None):
        '''
        filenames : list of path of filenames to the .mp3. Please note that separate the file names for train and test before calling this function.
        labels: the list of labels
        batch_size : scalar value typically 32 or 64 or power of 2
        '''
        self.x = filenames    
        self.y = labels
        self.label_dict = self.make_label_dict()
        self.batch_size = batch_size
        if shuffle:
            temp = list(zip(self.x,self.y))
            random.shuffle(temp)
            self.x,self.y = zip(*temp)
        self.duration = duration #in seconds
        self.resampling_rate = resampling_rate
        self.one_hot_labels = to_one_hot(self.y)
    
    def __getitem__(self,idx):
        batch_x_file_names = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_one_hot = self.one_hot_labels[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_x = []
        for idxx,name in enumerate(batch_x_file_names):
            clip,sr = librosa.load(name, sr=self.resampling_rate, duration=self.duration)
            if clip.shape[0]/self.resampling_rate < self.duration:
                clip = self.smart_append(clip,batch_y[idxx])
            S = np.abs(librosa.core.stft(clip, n_fft=2048, hop_length=None, win_length= None, window=signal.hamming))
            batch_x.append(S)
        return np.array(batch_x), np.array(batch_one_hot)
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def smart_append(self,clip,label):
        '''
        This will be called when any function with size less than duration is found
        '''
        while clip.shape[0]/self.resampling_rate < self.duration:
            #Randomly sample from the label dict
            fname = random.sample(self.label_dict[label],1)
            clip2,sr = librosa.load(*fname,sr=self.resampling_rate)
            clip = np.concatenate((clip,clip2))
        
        return clip[:self.duration*self.resampling_rate]
    
    def make_label_dict(self):
        label_dict = {}
        for filename,label in zip(self.x,self.y):
            label_dict[label] = label_dict.get(label,[]) + [filename]
        return label_dict

    
