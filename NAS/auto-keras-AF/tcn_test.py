import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)
rootPath = curPath
for i in range(2):
    rootPath = os.path.split(rootPath)[0]
print(rootPath)
sys.path.append(rootPath)

import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from utils.uts_classification.utils import readmts_uci_har,transform_labels
from utils.AFClassication.data import loaddata

import autokeras as ak
import numpy as  np
from NAS.logger import Logger
from tools import AdvancedLearnignRateScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 加载UCI_HAR_Dataset
(X_train, y_train), (Xval, yval), (final_testset, final_testtarget) , (R_train, Rval, Rtest), (
    P_train, Pval, Ptest), (Q_train, Qval, Qtest), (T_train, Tval, Ttest)= loaddata()


shape = X_train[0].shape
if len(shape) == 2:
    X_train[0] = np.expand_dims(X_train[0], axis=2)
    Xval[0] = np.expand_dims(Xval[0], axis=2)
    final_testset[0] = np.expand_dims(final_testset[0], axis=2)

NUM_CLASSES = 3

x_train = np.concatenate((X_train[0],Xval[0]),axis=0)
y_train = np.concatenate((y_train,yval),axis=0)
train_number = int(len(x_train) / 16) * 16


print(x_train.shape)
print(y_train.shape)
print(final_testset[0].shape)
print(final_testtarget.shape)
print(y_train[:3])

input_shape = x_train.shape[1:]
nb_classes = 3

