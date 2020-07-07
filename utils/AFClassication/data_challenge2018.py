# @Time : 2019/10/11 下午6:25 
# @Author : Xiaoyu Li
# @File : data.py 
# @Orgnization: Dr.Cubic Lab

import numpy as np
import scipy
import os
import pickle as dill


###########################
## Function to load data ##
###########################
def loaddata():

    data_path = '/data/weiyuhua/data/Challenge2018/preprocessed_data_new/'
    print("Loading data training set")
    with open(os.path.join(data_path, 'data_aug_train.pkl'), 'rb') as fin:
        res = dill.load(fin)
    X_ecg = res['trainset']
    y = res['traintarget']
    X_spec = res['trainsetSpec']

    with open(os.path.join(data_path, 'data_aug_val.pkl'), 'rb') as fin:
        res = dill.load(fin)
    val_set_ecg = res['val_set']
    val_set_spec = res['valsetSpec']
    val_target = res['val_target']

    with open(os.path.join(data_path, 'data_aug_test.pkl'), 'rb') as fin:
        res = dill.load(fin)
    final_testset_ecg = res['final_testset']
    final_testset_spec = res['testsetSpec']
    final_testtarget = res['final_testtarget']

    X_ecg = X_ecg.swapaxes(1,2)
    val_set_ecg = val_set_ecg.swapaxes(1,2)
    final_testset_ecg = final_testset_ecg.swapaxes(1,2)

    return (X_ecg, y), (val_set_ecg, val_target), (final_testset_ecg, final_testtarget)
    # return (X[0:100], y[0:100]),(val_set[0:100], val_target[0:100])

if __name__ == '__main__':
    a = np.arange(24).reshape((2, 3, 4))
    b = a.swapaxes(1,2)
    loaddata()
