import math
import numpy as np
import pandas as pd
import h5py
import os
import torch
import torch.nn.functional as F
import os.path as osp
import csv
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from SCCNet import *
from func import *


os.environ['CUDA_VISIBLE_DEVICES']='1'
ROOT = os.getcwd()
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
time_length = 384
class Time_Interval:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        #self.data = None
        #self.label = None
        #self.data_path = args.data_path
        #self.label_type = args.label_type
    def Run(self, threshold):

        if self.args.label_type == 'A':
            if self.args.testlist:
                loadpath = self.args.testlist_file
            else:
                loadpath = self.args.save_path + 'arousal/'
            savepath = self.args.save_path + 'arousal/time/' + self.args.retrain_type+'/'+str(threshold)+'/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        elif self.args.label_type == 'V':
            if self.args.testlist:
                loadpath = self.args.testlist_file
            else:
                loadpath = self.args.save_path + 'valence/'
            savepath = self.args.save_path + 'valence/time/' + self.args.retrain_type+'/'+str(threshold)+'/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)
        if self.args.retrain_type == 'nm':
            epochs = self.args.max_epoch
        if self.args.retrain_type == 'ft':
            epochs = self.args.ft_epoch
        intial_best_round = np.loadtxt(loadpath + "round.csv", delimiter=",")
        time_range = np.zeros(shape=[32,10,2])
        #loss_plot = np.zeros((2, 32, 10, 5, epochs)) # 0: training, 1: validating
        Aver_results = np.empty((2, 32, 10))# new model 0: loss acc, 1: loss balance_acc  #2: balance_loss acc #3: balance_loss balance_loss
        #valid_acc = np.zeros((32, 10, 5, epochs)) # 0: new model
        for sub in range(self.args.start_subjects, self.args.subjects):

            #valid_acc = np.zeros((2, 32, 10, epochs)) # 0: new model, 1: Fine-tune
            if self.args.label_type == 'V':
                data, label = load_per_subject_V(sub)
            elif self.args.label_type == 'A':
                data, label = load_per_subject_A(sub)
                
            testing_trial = np.empty(shape = [0, 4])
            loss_detail = np.empty(shape=[2, 32, 10, epochs])
            loss_detail1 = np.empty(shape=[2, 32, 10, epochs])
            
            savepath_sub = savepath + 'sub' + str(sub) + '/'
            if not os.path.exists(savepath_sub):
                os.makedirs(savepath_sub)

            with open(loadpath + 'sub_' + str(sub) + '/testing_trial.csv', newline='') as csvfile:
                rows = csv.reader(csvfile)
                for fold, row in enumerate(rows):
                    testing_trial = np.vstack([testing_trial, [int(float(i)) for i in row]])
            

            loadpath_sub = loadpath + 'sub_' + str(sub) + '/'
            prob_all = np.load(loadpath_sub + 'training_probability.npy')

            BEST_ROUND = np.zeros(shape=[10])
            for fold, row in enumerate(testing_trial):
                test_trial_l = [int(float(i)) for i in testing_trial[fold]]
                x_test, y_test = data[test_trial_l], label[test_trial_l]
                x_test, y_test = slidingWindows(x_test, y_test)
                x_test = torch.from_numpy(x_test).float().to(dev)
                y_test = torch.from_numpy(y_test).long().to(dev)
                test_dataset = Data.TensorDataset(x_test, y_test)
                testloader = Data.DataLoader(
                    dataset = test_dataset,
                    batch_size = 1,
                    shuffle = False,
                    num_workers = 0,
                )
                l = np.arange((40))
                
                l = np.delete(l, np.array(row).astype(int))
                tmp = prob_all[fold,l]
                fold_label = label[l]
                low = np.where(fold_label == 0)[0]
                low_trial_prob = tmp[low]
                low_avg = np.average(low_trial_prob, axis = 0)
                high = np.where(fold_label == 1)[0]
                high_trial_prob = tmp[high]
                high_avg = np.average(high_trial_prob, axis = 0)

                threshold = float(self.args.threshold/100)
                time_length = int(60*threshold)
                window_len = int(time_length-3)
                seg_cnt = int(58-window_len)
                winCnt = np.zeros(shape = [2, seg_cnt])
                for index in range(seg_cnt):
                    winCnt[0, index] =  np.sum(low_avg[index:(index+window_len)]) 
                    winCnt[1, index] =  np.sum(high_avg[index:(index+window_len)])
                time_range[sub, fold] = [np.argmax(winCnt[0]), np.argmax(winCnt[1])]
                low_data = data[low,:,:,np.argmax(winCnt[0])*128:(np.argmax(winCnt[0])+window_len+3)*128]
                high_data = data[high,:,:,np.argmax(winCnt[1])*128:(np.argmax(winCnt[1])+window_len+3)*128]
                x_train_fold = np.empty(shape=[0,1,32,(128*time_length)])
                y_train_flod = np.empty(shape=[0])
                x_train_fold = np.vstack([x_train_fold, low_data])
                y_train_flod = np.append(y_train_flod,fold_label[low])

                x_train_fold = np.vstack([x_train_fold, high_data])
                y_train_flod = np.append(y_train_flod, fold_label[high])
                
                BEST_ACC = 0.0
                for r in range(self.args.round):

                    x_train, y_train, x_valid, y_valid, weights = Split(x_train_fold, y_train_flod)
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

                    best_loss = 10000
                    if self.args.retrain_type == 'nm':
                        net = SCCNet().to(dev)
                    if self.args.retrain_type == 'ft':
                        net = torch.load(loadpath_sub + 'sub'+ str(sub) + '_' + str(fold) + '_' + str(int(intial_best_round[sub, fold])) +'.pt').to(dev)
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    optimizer = optim.Adam(net.parameters(),lr=1e-4, weight_decay=self.args.weight_decays)


                    # train
                    for e in range(epochs):
                        net.train()
                        running_loss = 0.0
                        for t, (xb, yb) in enumerate(trainloader):
                            pred = net(xb)
                            loss = criterion(pred, yb)
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            running_loss += loss.item()*128.0
                        #loss_plot[0, sub, fold, r, e] = (running_loss/len(trainloader))


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
                        #loss_plot[1, sub, fold, r, e] = (current_loss)
                        #valid_acc[sub, fold, r, e] = balanced_accuracy_score(labels, prediction)
                        


                        if current_loss < best_loss:
                            torch.save(net, savepath_sub + 'sub'+ str(sub) + '_' + str(fold) + '_' + str(r) +'.pt')
                            best_loss = current_loss
                            if balanced_accuracy_score(labels, prediction)>BEST_ACC:
                                BEST_ACC = balanced_accuracy_score(labels, prediction)
                                BEST_ROUND[fold] = r                
                            

                # test
                BEST_ROUND = BEST_ROUND.astype(int)
                    
                net = torch.load(savepath_sub + 'sub'+ str(sub) + '_' + str(fold) + '_' + str(BEST_ROUND[fold]) +'.pt').to(dev)
                net.eval()
                acc = 0
                prediction = []
                labels = []
                for xb, yb in testloader:
                    pred = net(xb)
                    prediction.append(pred.argmax().item())
                    labels.append(yb.item())

                balance_ACC = balanced_accuracy_score(labels, prediction)
                Aver_results[0, sub, fold] = balance_ACC
                Aver_results[1, sub, fold] = accuracy_score(labels, prediction)



                #last_acc[sub, fold_, Round] = acc
                #last_f1[sub, fold_, Round] = f1
                
            np.savetxt(savepath_sub + 'round.csv', BEST_ROUND.astype(int), delimiter=",")

        #np.save(savepath + 'Results_with_Round', retsult_with_round)
        np.save(savepath + 'time_range', time_range)
        Aver = np.mean(Aver_results, axis = 2)
        AVG_acc = pd.DataFrame(Aver_results[0])
        AVG_f1 = pd.DataFrame(Aver_results[1])
        AVG = pd.DataFrame(Aver)
        file_path_xlsx = savepath +"testing_result.xlsx"
        writer = pd.ExcelWriter(file_path_xlsx)
        AVG_acc.to_excel(writer, '(loss)Acc')
        AVG_f1.to_excel(writer, '(loss)balanced_acc')
        AVG.to_excel(writer, 'Compare')
        writer.save()
        

        #pd_acc.to_excel(writer, 'full', index=False)

        #writer.save()


