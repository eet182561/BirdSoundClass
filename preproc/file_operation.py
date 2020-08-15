# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 13:00:43 2020

@author: itsyo
"""


import csv
import pandas as pd
import numpy as np
from scipy.io import wavfile
import wave
import os
import shutil


'''
input: read csv file at aparticular path, delimiter used in the csv file
output: list of [birdtype, filename] where secondary_labels is []
desc: ectract re;evant colums of the train file
'''
def read_csv_file(path='E:\\database\\kaggle\\bird_sound\\trainCopy.csv',csv_delimiter = ','):
    with open(path, encoding = 'cp850') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=csv_delimiter)
        data_list = list(csv_reader)
        data_list = data_list[1:]
    extracted_data = []
    for data in data_list:
        # print([data[2],data[7]])
        if data[12] =='[]':
            extracted_data.append([data[2],data[7]])
    return extracted_data

'''
input: output csv file name, delimiter
output: write a csv file 
desc: write a csv file of the data sent in the list
'''
def write_csv_file(path,  data, csv_delimiter = ','):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=csv_delimiter)
        writer.writerows(data)



# data = pd.read_csv('E:\\database\\kaggle\\bird_sound\\trainCopy.csv', usecols=['ebird_code', 'filename', 'secondary_labels']) 
# data_list = [data['ebird_code'], data['filename'], data['secondary_labels']]
# path='E:\\database\\kaggle\\bird_sound\\trainCopy.csv'
# csv_delimiter = ','
# a = read_csv_file()
# write_csv_file('C:\\Users\\itsyo\\BirdSoundClass\\rough_work\\sample.csv', a)