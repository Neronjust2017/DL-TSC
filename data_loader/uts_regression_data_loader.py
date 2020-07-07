# -*- coding:utf-8 -*-
from base.base_data_loader import BaseDataLoader
import numpy as np
import pandas as pd
import tensorflow as tf
import math

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class UtsRegressionDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(UtsRegressionDataLoader, self).__init__(config)
        self.w = config.args.window
        self.h = config.args.horizon
        self.batch_size = config.args.batchSize
        self.shuffle = True
        self.n = 0


        if config.args.airData == True:
            raw_data = pd.read_csv('./datasets/regression/air/res_mean.csv')
            raw_data = raw_data._values
            raw_data = np.array(raw_data)
            self.rawdat = raw_data
        else:
            fin = open(config.args.dataPath)
            self.rawdat = np.loadtxt(fin, delimiter=',')
            fin.close()

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.scale = np.ones(self.m)
        self._normalized(config.args.normalize)
        instance_num = self.n - self.w - self.h + 1
        indexes = [i for i in range(instance_num)]
        np.random.shuffle(indexes)
        cut1 = int(instance_num * config.args.trainRatio)
        cut2 = int(instance_num * (config.args.trainRatio + config.args.validRatio))
        self.train_set_index, self.valid_set_index, self.test_set_index = indexes[0:cut1], indexes[cut1:cut2], indexes[cut2:]
        self.list_IDs = indexes


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

    def get_train_data(self):
        return DataGenerator(self.dat, self.train_set_index, self.w, self.h, self.m, self.batch_size, shuffle=True)


    def get_valid_data(self):
        return DataGenerator(self.dat, self.valid_set_index, self.w, self.h, self.m, self.batch_size, shuffle=True)

    def get_test_data(self):
        return DataGenerator(self.dat, self.test_set_index, self.w, self.h, self.m, self.batch_size, shuffle=True)

    def get_inputshape(self):
        return self.batch_size, self.w, self.m


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