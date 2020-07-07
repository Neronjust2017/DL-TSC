from base.base_data_loader import BaseDataLoader
from utils.uts_classification.utils import readucr,readmts,transform_labels,readmts_uci_har,readmts_ptb,readmts_ptb_aug
import sklearn
import numpy as np
import os
import pickle as dill
from collections import Counter

class UtsClassificationDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(UtsClassificationDataLoader, self).__init__(config)

        if config.dataset.type == 'uts':
            if config.dataset.name == 'AFClassification':
                from utils.AFClassication.data import loaddata
                (X_train, y_train), (Xval, yval), (final_testset, final_testtarget), (R_train, Rval, Rtest), (
                    P_train, Pval, Ptest), (Q_train, Qval, Qtest), (T_train, Tval, Ttest) = loaddata()
                X_train = X_train[0]
                X_val = Xval[0]
                y_val = yval
                X_test = final_testset[0]
                y_test = final_testtarget
                self.nb_classes = 3
                self.y_train = y_train
                self.y_test = y_test
                self.X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                self.y_val = y_val
                self.y_true = np.argmax(y_test, axis=1)

            elif config.dataset.name == 'ptbdb':

                file_path = './datasets/uts_data/ptbdb/'
                X_train, y_train, X_val, y_val, X_test, y_test = readmts_ptb_aug(file_path)
                self.nb_classes = len(np.unique(np.concatenate((y_train, y_val, y_test), axis=0)))
                y_train,  y_val, y_test = transform_labels(y_train, y_test, y_val)
                self.X_val = X_val.reshape((self.X_val.shape[0], self.X_val.shape[1], 1))
                enc = sklearn.preprocessing.OneHotEncoder()
                enc.fit(np.concatenate((y_train, y_val, y_test), axis=0).reshape(-1, 1))
                self.y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
                self.y_val = enc.transform(self.y_val.reshape(-1, 1)).toarray()
                self.y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
            else:
                file_name = 'datasets/uts_data/' + config.dataset.name + '/' + config.dataset.name
                X_train, y_train = readucr(file_name + '_TRAIN.txt')
                X_test, y_test = readucr(file_name + '_TEST.txt')
                self.nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
                # make the min to zero of labels
                y_train, y_test = transform_labels(y_train, y_test)

        else:
            if config.dataset.name == 'UCI_HAR_Dataset':
                file_name = 'datasets/mts_data/' + config.dataset.name
                X_train, y_train, X_test, y_test = readmts_uci_har(file_name)
                # 调整划分比例
                data = np.concatenate((X_train, X_test),axis=0)
                label = np.concatenate((y_train, y_test),axis=0)
                N = data.shape[0]
                ind = int(N*0.9)
                X_train = data[:ind]
                y_train = label[:ind]
                X_test = data[ind:]
                y_test = label[ind:]
                self.nb_classes = 6
                # make the min to zero of labels
                y_train, y_test = transform_labels(y_train, y_test)

            elif config.dataset.name == 'Challeng2018':
                from utils.AFClassication.data_challenge2018 import loaddata
                (X_train, y_train), (Xval, yval), (final_testset, final_testtarget)= loaddata()
                X_val = Xval
                X_test = final_testset
                y_val = yval
                y_test = final_testtarget
                self.nb_classes = 9
                self.X_val = X_val
                self.y_train = y_train
                self.y_test = y_test
                self.y_val = y_val
                self.y_true = np.argmax(y_test, axis=1)
            else:
                file_name = 'datasets/mts_data/' + config.dataset.name + '/' + config.dataset.name
                X_train, y_train, X_test, y_test, self.nb_classes = readmts(file_name)

        if config.dataset.name not in ['ptbdb','AFClassification', 'Challeng2018']:
            # save orignal y because later we will use binary
            self.y_true = y_test.astype(np.int64)
            # transform the labels from integers to one hot vectors
            enc = sklearn.preprocessing.OneHotEncoder()
            enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
            self.y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
            self.y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        if config.dataset.type == 'uts':
            self.X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            self.X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        else:
            self.X_train = X_train
            self.X_test = X_test
        self.train_size = self.X_train.shape[0]
        self.test_size = self.X_test.shape[0]
        self.input_shape = self.X_train.shape[1:]

        if(self.config.model.name == "tlenet"):
            from models.classification.tlenet import Classifier_TLENET
            self.X_train, self.y_train, self.X_test, self.y_test, self.tot_increase_num, \
            self.input_shape, self.nb_classes = Classifier_TLENET().pre_processing(self.X_train, self.y_train, self.X_test, self.y_test)
            print(self.input_shape)
            print("********************************")

    def get_train_data(self):
        if self.config.dataset.name in ['ptbdb','Challeng2018']:
            return self.X_train, self.y_train, self.X_val, self.y_val
        else:
            return self.X_train, self.y_train


    def get_test_data(self):
        if (self.config.model.name == "tlenet"):
            return self.X_test, self.y_test, self.y_true, self.tot_increase_num
        return self.X_test, self.y_test, self.y_true

    def get_inputshape(self):
        return self.input_shape

    def get_nbclasses(self):
        return self.nb_classes

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size