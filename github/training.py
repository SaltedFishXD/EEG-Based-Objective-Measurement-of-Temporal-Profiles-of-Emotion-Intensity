import numpy as np
import pandas as pd
import datetime
import os
import csv
import h5py
import copy

import random
import os.path as osp
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from SCCNet import *
from func import *
time_length = 384


class training:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        #self.data = None
        #self.label = None
        #self.data_path = args.data_path
        #self.label_type = args.label_type
        
        

    def Run_CrossTrial(self, start, end, round, Epochs, type, weight_decays, filename, testlist, testlist_file):
        os.environ['CUDA_VISIBLE_DEVICES']= self.args.gpu
        ROOT = os.getcwd()
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type == 'A':
            savepath = filename+'arousal/'
        elif type == 'V':
            savepath = filename+'valence/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        print(dev)


        loss_plot = np.zeros((2, 32, 10, round, Epochs)) # 0: training, 1: validating
        test_acc = np.zeros((32, 10, round,Epochs)) # 0: acc
        valid_acc = np.zeros((32, 10, round, Epochs)) # 0: acc
        retsult_with_round = np.zeros((2, 32, 10)) 
        Aver_results = np.empty((2, 32))# 0: acc, 1: F1-score
        BEST_ROUND = np.zeros(shape=[32, 10])
        for sub in range(start, end):
        #for sub in sub_list:
            savepath_sub = savepath + 'sub_' + str(sub) + '/'
            if not os.path.exists(savepath_sub):
                os.makedirs(savepath_sub)
            test_list = np.zeros(shape=[10, 4])
            if type == 'V':
                data, label = load_per_subject_V(sub)
            else:
                data, label = load_per_subject_A(sub)

            if testlist:
                with open(testlist_file + 'sub_' + str(sub) + '/testing_trial.csv', newline='') as csvfile:
                    rows = csv.reader(csvfile)

                    for fold, row in enumerate(rows):
                        test_list[fold] = [int(float(i)) for i in row]
            #test_list = KFOLD(label, 10)
            else:
                test_list = KFOLD(label, 10)
            test_list = test_list.astype(int)
            train_list = np.zeros(shape=[10, 36])
            for fold_ in range(10):
                train_index = np.arange((40))
                train_index = np.delete(train_index, np.array(test_list[fold_]).astype(int))
                train_list[fold_] = train_index
                x_test, y_test = data[test_list[fold_]], label[test_list[fold_]]
                x_test, y_test = slidingWindows(x_test, y_test)
                
                Sub_acc = 0.0
                Sub_f1 = 0.0

                x_test = torch.from_numpy(x_test).float().to(dev)
                y_test = torch.from_numpy(y_test).long().to(dev)

                test_dataset = Data.TensorDataset(x_test, y_test)

                testloader =  Data.DataLoader(
                        dataset = test_dataset,
                        batch_size = 1,
                        shuffle = False,
                        num_workers = 0,
                )

                BEST_ACC = np.zeros(shape=[5])
                for Round in range(round):
                    x_train, y_train = data[train_index], label[train_index]

                    x_train, y_train, x_valid, y_valid, weights = Split(x_train, y_train)
                    x_train, y_train = slidingWindows(x_train, y_train)
                    x_valid, y_valid = slidingWindows(x_valid, y_valid)
                    

                    class_weights = torch.FloatTensor(weights).to(dev)

                    x_train = torch.from_numpy(x_train).float().to(dev)
                    y_train = torch.from_numpy(y_train).long().to(dev)
                    x_valid = torch.from_numpy(x_valid).float().to(dev)
                    y_valid = torch.from_numpy(y_valid).long().to(dev)

                    train_dataset = Data.TensorDataset(x_train, y_train)
                    valid_dataset = Data.TensorDataset(x_valid, y_valid)
                    
                    trainloader = Data.DataLoader(
                        dataset = train_dataset,
                        batch_size = 128,
                        shuffle = True,
                        num_workers = 0,
                    )
                    validloader = Data.DataLoader(
                        dataset = valid_dataset,
                        batch_size = 1,
                        shuffle = False,
                        num_workers = 0,
                    )

                    last_loss = 10000
                    best_loss = 10000
                    max_patience = 5
                    patience = 0
                    save_epoch = 0
                    best_ACC = 0
                    best_epoch = 0

                    #net = EEGNet_DEAP().to(dev)
                    net = SCCNet().to(dev)
                    #net = ShallowConvNet().to(dev)
                    #criterion = nn.CrossEntropyLoss()
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    optimizer = optim.Adam(net.parameters(),lr=1e-4, weight_decay=weight_decays)


                    # train
                    for epochs in range(Epochs):
                        net.train()
                        running_loss = 0.0
                        for t, (xb, yb) in enumerate(trainloader):
                            pred = net(xb)
                            loss = criterion(pred, yb)
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            running_loss += loss.item()*128.0
                        loss_plot[0, sub, fold_, Round, epochs] = (running_loss/len(trainloader))


                        # early stop
                        net.eval()
                        current_loss = 0
                        correct = 0
                        prediction = []
                        labels = []
                        for xb, yb in validloader:
                            #with torch.no_grad():
                            pred = net(xb)
                            loss = criterion(pred, yb)
                            prediction.append(pred.argmax().item())
                            labels.append(yb.item())
                            current_loss += loss
                        loss_plot[1, sub, fold_, Round, epochs] = (current_loss)
                        valid_acc[sub, fold_, Round, epochs] = balanced_accuracy_score(labels, prediction)
                        


                        if current_loss < best_loss:
                            torch.save(net, savepath_sub + 'sub'+ str(sub) + '_' + str(fold_) + '_' + str(Round) +'.pt')
                            BEST_ACC[Round] = balanced_accuracy_score(labels, prediction)
                        
                BEST_ROUND[sub, fold_] = np.argmax(BEST_ACC)



                # test
                BEST_ROUND = BEST_ROUND.astype(int)               
                net = torch.load(savepath_sub + 'sub'+ str(sub) + '_' + str(fold_) + '_' + str(BEST_ROUND[sub, fold_]) +'.pt').to(dev)
                net.eval()
                acc = 0
                prediction = []
                labels = []
                for xb, yb in testloader:
                    pred = net(xb)
                    prediction.append(pred.argmax().item())
                    labels.append(yb.item())

                balance_ACC = balanced_accuracy_score(labels, prediction)
                retsult_with_round[0, sub, fold_] = balance_ACC
                retsult_with_round[1, sub, fold_] = accuracy_score(labels, prediction)



                #last_acc[sub, fold_, Round] = acc
                #last_f1[sub, fold_, Round] = f1
                
            np.savetxt(savepath_sub + 'testing_trial.csv', test_list.astype(int), delimiter=",")
            np.savetxt(savepath_sub + 'train_trial.csv', train_list.astype(int), delimiter=",")

        np.savetxt(savepath + 'round.csv', BEST_ROUND.astype(int), delimiter=",")
        np.save(savepath+'loss', loss_plot)
        np.save(savepath + 'Process_result_test', test_acc)
        np.save(savepath + 'Process_result_val', valid_acc)
        np.save(savepath + 'Results_with_Round', retsult_with_round)
        Aver_results = np.mean(retsult_with_round, axis = 2)
        if type == 'V':
            writer = pd.ExcelWriter(savepath + 'Valence_results.xlsx')
        else:
            writer = pd.ExcelWriter(savepath + 'Arousal_results.xlsx')
        pd_acc = pd.DataFrame(Aver_results[0])
        pd_acc1 = pd.DataFrame(Aver_results[1])
        

        pd_acc.to_excel(writer, 'balalced_ACC', index=False)
        pd_acc1.to_excel(writer, 'Acc', index=False)

        writer.save()



