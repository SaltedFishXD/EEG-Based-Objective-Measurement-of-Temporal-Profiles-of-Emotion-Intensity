from cmath import nan
import numpy as np
import pandas as pd
import datetime
import os
import csv
import h5py
import copy
import time

import random
import os.path as osp
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score
from scipy.signal import butter, lfilter
from sklearn import preprocessing

time_length = 384

def load_per_subject_A(sub):
    save_path = os.getcwd()
    data_type = './data_raw_A/'
    sub_code = 'sub' + str(sub) + '.hdf'
    path = osp.join(data_type, sub_code)
    dataset = h5py.File(path, 'r')
    data = np.array(dataset['data'])
    label = np.array(dataset['label'])
    return data, label

def load_per_subject_V(sub):
    save_path = os.getcwd()
    data_type = './data_raw_V/'
    sub_code = 'sub' + str(sub) + '.hdf'
    path = osp.join(data_type, sub_code)
    dataset = h5py.File(path, 'r')
    data = np.array(dataset['data'])
    label = np.array(dataset['label'])
    return data, label



def slidingWindows(data, label):
    
    lenght = data.shape
    Data = data[:,:,:,0:time_length]
    Label = label
    start = 128
    while 1:
        tmp = data[:,:,:,start:start+time_length]
        Data = np.concatenate((Data, tmp), axis=0)
        Label = np.concatenate((Label, label))
        start += 128
        if start + time_length >= lenght[3]:
            tmp = data[:,:,:,lenght[3]-time_length:lenght[3]]
            Data = np.concatenate((Data, tmp), axis=0)
            Label = np.concatenate((Label, label))
            break
    return Data, Label


def split_train_valid_set(x_train, y_train, num):
    shuffler = np.random.permutation(len(x_train))
    #print(shuffler)
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    n = int(len(x_train) / num)
    return x_train[n:], y_train[n:], x_train[:n], y_train[:n], shuffler[:n]


def Count_class_num(label):
    Num_low = np.where(label == 0)[0]
    Num_high = np.where(label == 1)[0]
    if len(Num_low) == 0 or len(Num_high) == 0 :
        return False, len(Num_low), len(Num_high)
    else:
        return True, len(Num_low), len(Num_high)

def KFOLD(label, fold):
    return_list = np.zeros(shape=[10, 4])
    low = np.where(label==0)[0]
    high = np.where(label==1)[0]

    low = np.random.permutation(low)
    high = np.random.permutation(high)

    low_array = np.array_split(low, fold)
    high_array = np.array_split(high, fold)
    for i in range(fold):
        return_list[i] = np.append(low_array[i], high_array[9-i])
    return return_list


def Split_retrain(data, label):
    low = np.where(label==0)[0]
    high = np.where(label==1)[0]
    low = np.random.permutation(low)
    high = np.random.permutation(high)
    num = (low.shape[0]+high.shape[0])//10
    valid_index = np.zeros(shape=[0])
    valid_index = np.append(valid_index, low[:num])
    valid_index = np.append(valid_index, high[:num])
    train_index = np.zeros(shape=[0])
    train_index = np.append(train_index, low[num:])
    train_index = np.append(train_index, high[num:])
    train_index = train_index.astype(int)
    valid_index = valid_index.astype(int)
    x_train, y_train = data[train_index], label[train_index]
    x_valid, y_valid = data[valid_index], label[valid_index]
    
    TF, a, b = Count_class_num(y_train)
    return x_train, y_train, x_valid, y_valid, [1-(a/(a+b)), 1-(b/(a+b))]

def Split(data, label):
    low = np.where(label==0)[0]
    high = np.where(label==1)[0]
    low = np.random.permutation(low)
    high = np.random.permutation(high)
    valid_index = np.zeros(shape=[0])
    valid_index = np.append(valid_index, low[:2])
    valid_index = np.append(valid_index, high[:2])
    train_index = np.zeros(shape=[0])
    train_index = np.append(train_index, low[2:])
    train_index = np.append(train_index, high[2:])
    train_index = train_index.astype(int)
    valid_index = valid_index.astype(int)
    x_train, y_train = data[train_index], label[train_index]
    x_valid, y_valid = data[valid_index], label[valid_index]
    
    TF, a, b = Count_class_num(y_train)
    return x_train, y_train, x_valid, y_valid, [1-(a/(a+b)), 1-(b/(a+b))]

    if a >= b:
        return x_train, y_train, x_valid, y_valid, [1, a/b]
    else:
        return x_train, y_train, x_valid, y_valid, [b/a, b/b]

def Normalized_type1(data):
    shape = data.shape
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for i in range(shape[0]):
        data[i,0] = scaler.fit_transform(data[i, 0])
    #print(data.shape)
    return data

def CreateDetail(savepath, baseline, start, end, round, Epochs, type, weight_decays, normalized, Validation):
    seconds = time.time()
    local_time = time.ctime(seconds)
    lr = ['default', 'None Update', 'Validation Update Ver 1', 'Validation Update Ver 2', 'Test Update Ver 3']
    Normal_type = ['None', 'Normalize Trial', 'Normalize windows']
    Baseline_type = ['None', "Average Baseline Remove", '3s Baseline window remove']
    Validation_list = ['Cross Window', 'Cross Trial']
    f = open(savepath + "detail.txt", "w")
    f.write(f"Date Time\t{local_time}\n")
    f.write(f'Start subject: {str(start)} \n End subject: {str(end)}\n')
    f.write(f'Round: {str(round)}\n Epochs:{str(Epochs)}\n')
    f.write(f'Baseline: {Baseline_type[baseline]}\n')
    f.write(f'Learning rate: {lr[type]}\n')
    f.write(f'Weight Decays: {str(weight_decays)}\n')
    f.write(f'Normalize type: {Normal_type[normalized]}\n')
    f.write(f'Validation type: {Validation_list[Validation]}\n')
    f.close()
