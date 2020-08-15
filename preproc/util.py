# -*- coding: utf-8 -*-
import numpy as np
"""
Created on Sat Aug 15 22:30:09 2020

@author: itsyo
"""


'''
input: bird data list
output: total no of bird classes, bird name and label, bird data name list
desc: take the list of the training wav file name and bird name and used to generate the label corrosponding to the bird
'''
def assign_bird_label(bird_data_list): # returns [total_speaker, labels,labelled_speaker_list]
    data = np.array(bird_data_list)
    bird_data_set = set(data[:, 0])  # extrating just speakers
    label_index =0
    bird_label = {}
    for bird_data in bird_data_set: # assigning labels to every speaker
        bird_label.update({bird_data : label_index})
        label_index = label_index + 1
    #print(speaker_label)
    # creating a updating label to main data list
    for detail in bird_data_list:
        print(detail[0])
        detail.append(bird_label.get(detail[0]))
    
    
    return [len(bird_data_set), bird_label, bird_data_list]