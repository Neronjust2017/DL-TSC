# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import math

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train_ratio, valid_ratio, args):
        self.w = args.window
        self.h = args.horizon

        if args.forAir == True:
            raw_data = pd.read_csv('./data/air/res_mean.csv')
            raw_data = raw_data._values
            # random.shuffle(raw_data)
            raw_data = np.array(raw_data)
            self.rawdat = raw_data
            self.dat = np.zeros(self.rawdat.shape)
            self.n, self.m = self.dat.shape
            self.scale = np.ones(self.m)
            self._normalized(args.normalize)
            instance_num = self.n - self.w - self.h + 1
            indexes = [i for i in range(self.n - self.w - self.h + 1)]

            # 此处存疑， RNN模型中不应该shuffle
            np.random.shuffle(indexes)

            cut1 = int(instance_num * train_ratio)
            cut2 = int(instance_num * (train_ratio + valid_ratio))
            self.train_set_index, self.valid_set_index, self.test_set_index = indexes[0:cut1], indexes[cut1:cut2], indexes[cut2:]
        else:
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
            self.dat = np.zeros(self.rawdat.shape)
            self.n, self.m = self.dat.shape
            self.scale = np.ones(self.m)
            self._normalized(args.normalize)
            instance_num = self.n - self.w - self.h + 1
            indexes = [i for i in range(self.n - self.w - self.h + 1)]

            # 此处存疑， RNN模型中不应该shuffle
            np.random.shuffle(indexes)

            cut1 = int(instance_num * train_ratio)
            cut2 = int(instance_num * (train_ratio + valid_ratio))
            self.train_set_index, self.valid_set_index, self.test_set_index = indexes[0:cut1], indexes[cut1:cut2], indexes[cut2:]
            fin.close()

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

class DataGenerator(tf.compat.v2.keras.utils.Sequence):

    def __init__(self, data, indexes, window_size, horizon, channel_num, batch_size, shuffle=True):
        self.w = window_size
        self.h = horizon
        self.m = channel_num
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = indexes
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0

        return data

    def __len__(self):
        # Return the number of batches of the dataset
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:
                               (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self._generate_batch(list_IDs_temp)
        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_batch(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.w, self.m))
        y = np.empty((self.batch_size, self.m))
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.data[ID:ID + self.w, :]
            y[i,] = self.data[ID + self.w, :]
        return X, y