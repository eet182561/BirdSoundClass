# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:30:09 2020

@author: itsyo
"""


'''
input: bird data list
output: total no of bird classes, bird name and label, bird data name list
desc: take the list of the training wav file name and bird name and used to generate the label corrosponding to the bird
'''
def assign_speaker_label(speaker_data): # returns [total_speaker, labels,labelled_speaker_list]
    data = np.array(speaker_data)
    speaker_list = set(data[:, 1])  # extrating just speakers
    label_index =0
    speaker_label = {}
    for speaker in speaker_list: # assigning labels to every speaker
        speaker_label.update({speaker : label_index})
        label_index = label_index + 1
    #print(speaker_label)
    # creating a updating label to main data list
    for detail in speaker_data:
        detail.append(speaker_label.get(detail[1]))
    
    
    return [len(speaker_list), speaker_label, speaker_data]
