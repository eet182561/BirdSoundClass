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
    def __init__(self,filenames,labels,batch_size=32,duration=10,resampling_rate=40000,shuffle = True,small_files = None):
        '''
        filenames : list of path of filenames to the .mp3. Please note that separate the file names for train and test before calling this function.
        labels: the list of labels
        batch_size : scalar value typically 32 or 64 or power of 2
        '''
        self.x = filenames
        if small_files is None:
            self.small_files_index = self.calculate_small_files()
        else:
            with open(small_files,'rb') as f:
                self.small_files_index = pickle.load(f)
            
        self.y = labels
        self.label_dict = self.make_label_dict()
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
        for idx,name in enumerate(batch_x_file_names):
            clip = librosa.load(name, sr=self.resampling_rate, duration=self.duration)
            if clip.shape[0]/self.resampling_rate < self.duration:
                clip = self.smart_append(clip,batch_y[idx])
            S = np.abs(librosa.core.stft(clip_res))
            batch_x.append(S)
        return batch_x, np.array(batch_y)
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def smart_append(self,clip,label):
        '''
        This will be called when any function with size less than duration is found
        '''
        while clip.shape[0]/self.resampling_rate < self.duration:
            #Randomly sample from the label dict
            fname = random.sample(self.label_dict[label],1)
            clip2 = librosa.load(*fname,sr=self.resampling_rate)
            clip = np.concatenate(clip,clip2)
        
        return clip[:self.duration*self.resampling_rate]
    
    def make_label_dict(self):
        lebel_dict = {}
        for filename,label in zip(self.x,self.y):
            label_dict[label] = label_dict.get(label,default=[]) + filename
        return label_dict

    #The below function is not required
    def calculate_small_files(self):
        '''
        Dont't know how much time it will take. So I will write the indexes which can be downloaded.
        '''
        indexs = []
        for idx,file in enumerate(self.x):
            clip,sr = librosa.load(file,sr=None)
            #duration = librosa_get_duration(clip,sr)
            if duration < self.duration :
                indexs.append(idx)
            del clip
            del sr
            del duration
        filename = 'samallfiles_'+str(self.duration)+'.pkl'
        with open(filename,'wb') as f:
            pickle.dump(indexs,f)
        return indexs
