
import copy
import _pickle as cPickle
import os
import numpy as np
import os.path as osp
import h5py

class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.data_path = args.data_path
        self.label_type = args.label_type

    def run(self, num_subject):
        # this is the main function to prepare the data
        # finial output: data(trial x 1 x channel x data)
        #                label(trial x 1)
        # the output will be saved as sub0.hdf located at different folders of each type of data
        # num_subject : int, controls how many subjects are loaded

        trial_total = []
        for sub in range(num_subject):
            data_, label_ = self.load_data_per_subject(sub)
            # select label type here
            label_ = self.label_selection(label_)
            # expand one dimension for deep learning
            data_ = np.expand_dims(data_, axis=1)
            print('Data and label prepared!')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

    # Load the data subject by subject
    def load_data_per_subject(self, sub):
        sub += 1
        if (sub < 10):
            sub_code = str('s0' + str(sub) + '.dat')
        else:
            sub_code = str('s' + str(sub) + '.dat')

        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        data = subject['data'][:, 0:32, 3 * 128:]  # Excluding the first 3s of baseline
        #   data: 40 x 32 x 7680
        #   label: 40 x 4
        channels = self.load_channel_order()
        data_temp = np.stack([data[:, chan, :] for chan in channels], axis=1)
        data = data_temp
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def load_channel_order(self):
        ch_order = []
        file_name = 'channel_baseline.txt'
        with open(file_name, "r") as f:
            ch_order = [int(l.strip())-1 for l in f]
        return ch_order

    def label_selection(self, label):
        if self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'V':
            label = label[:, 0]
        elif self.label_type == 'L':
            label = label[:, 3]
        label = np.where(label <= 5, 0, label)
        label = np.where(label > 5, 1, label)
        print('Binary label generated!')
        return label

    def save(self, data, label, sub):
        save_path = os.getcwd()
        data_type = 'data_' + self.args.data_format + '_' + self.args.label_type
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

