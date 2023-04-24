import math
import numpy as np
import pandas as pd
import h5py
import os
import torch
import torch.nn.functional as F
import os.path as osp
from sklearn.metrics import f1_score, accuracy_score

from SCCNet import *
from func import *

import csv
os.environ['CUDA_VISIBLE_DEVICES']='1'
ROOT = os.getcwd()
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
time_length = 384

class Profile:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        #self.data = None
        #self.label = None
        #self.data_path = args.data_path
        #self.label_type = args.label_type

    def generate(self, start, end):
        if self.args.testlist:
            loadpath = self.args.testlist_file
            savepath = self.args.testlist_file
        else:
            if self.args.label_type == 'V':
                loadpath = self.args.save_path+'valence/'
                savepath = self.args.save_path+'valence/'
            if self.args.label_type == 'A':
                loadpath = self.args.save_path+'arousal/'
                savepath = self.args.save_path+'arousal/'
        best_round = np.loadtxt(loadpath + "round.csv", delimiter=",")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        #print(savepath)

        for sub in range(start, end):
            #print(sub)
            training_window = np.zeros(shape=[10, 40, 58, 1 ,32, 384])
            if self.args.label_type == 'V':
                training_probability = np.zeros(shape=[10,40,58])
                data, label = load_per_subject_V(sub)
            elif self.args.label_type == 'A':
                training_probability = np.zeros(shape=[10,40,58])
                #training_probability = np.zeros(shape=[10,5,40,58])
                data, label = load_per_subject_A(sub)


            savepath_sub = savepath + 'sub_' + str(sub) + '/'

            if not os.path.exists(savepath_sub):
                os.makedirs(savepath_sub)
            
            testing_trial = np.empty(shape = [0, 4])

            with open(loadpath + 'sub_' + str(sub) + '/testing_trial.csv', newline='') as csvfile:
                rows = csv.reader(csvfile)
                for fold, row in enumerate(rows):
                    testing_trial = np.vstack([testing_trial, [int(float(i)) for i in row]])
                #print(testing_trial)
                for fold, row in enumerate(testing_trial):
                    index = np.arange((40))
                    
                    index = np.delete(index, np.array(row).astype(int))
                    #print(index.shape)
                    
                    for trial in index:
                        
                        x_train, y_train = data[trial], label[trial]
                        trial_label = label[trial]
                        #print(f'Sub {sub} Fold {fold}')
                        x_train = np.expand_dims(x_train, axis = 0)
                        x_train, y_train = slidingWindows(x_train, np.array([y_train]))
                        training_window[fold, trial] = x_train

                        x_train = torch.from_numpy(x_train).float().to(dev)
                        y_train = torch.from_numpy(y_train).long().to(dev)
                        train_dataset = Data.TensorDataset(x_train, y_train)
                        #print(x_train.shape, y_train.shape)
                        trainloader = Data.DataLoader(
                            dataset = train_dataset,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0,
                        )
                        if self.args.label_type == 'V':
                            #for Round in range(5):
                            net = torch.load(loadpath + 'sub_'+ str(sub)+ '/' +  'sub'+ str(sub)+ '_' + str(fold) + '_' + str(int(best_round[sub, fold])) + '.pt').to(dev)
                            net.eval()
                            for window, (xb, yb) in enumerate(trainloader):
                                pred = net(xb)
                                prob = F.softmax(pred, dim=1)
                                prob = prob.data.tolist()[0]
                                if trial_label:
                                    prob = np.array(prob)[1]
                                else:
                                    prob = np.array(prob)[0]
                                #print(window, prob)
                                training_probability[fold, trial, window] = prob
                        if self.args.label_type == 'A':
                            #round_list = np.load(loadpath + 'Results_with_Best_Round.npy')
                            #for Round in [round_list[0, sub, fold]]:
                            #for Round in range(5):
                            net = torch.load(loadpath + 'sub_'+ str(sub)+ '/' +  'sub'+ str(sub)+ '_' + str(fold) + '_' + str(int(best_round[sub, fold])) + '.pt').to(dev)
                            net.eval()
                            for window, (xb, yb) in enumerate(trainloader):
                                pred = net(xb)
                                prob = F.softmax(pred, dim=1)
                                prob = prob.data.tolist()[0]
                                if trial_label:
                                    prob = np.array(prob)[1]
                                else:
                                    prob = np.array(prob)[0]
                                #print(window, prob)
                                #training_probability[fold, 0, trial, window] = prob
                                #print(window, prob)
                                training_probability[fold, trial, window] = prob
                    
            
            #np.save(savepath_sub + 'training_window', training_window)
            np.save(savepath_sub + 'training_probability', training_probability)
                
                    
                        
                        
                    

