# @Time : 2019/10/11 下午6:25 
# @Author : Xiaoyu Li
# @File : data.py 
# @Orgnization: Dr.Cubic Lab

import numpy as np
import scipy.io

###########################
## Function to load data ##
###########################
def loaddata():
    '''
        Load training/test data into workspace

        This function assumes you have downloaded and padded/truncated the
        training set into a local file named "trainingset.mat". This file should
        contain the following structures:
            - trainset: NxM matrix of N ECG segments with length M
            - traintarget: Nx4 matrix of coded labels where each column contains
            one in case it matches ['A', 'N', 'O', '~'].

    '''
    print("Loading data training set")
    matfile = scipy.io.loadmat('/data/weiyuhua/data/AF_preprocessed_data/data_aug_1.mat')
    matfile2 = scipy.io.loadmat('/data/weiyuhua/data/AF_preprocessed_data/data_aug_2.mat')
    X1_spec = matfile['trainsetSpec']
    y1 = matfile['traintarget']
    X2_spec = matfile2['trainsetSpec']
    y2 = matfile2['traintarget']
    X1_ecg = matfile['trainset']
    X2_ecg = matfile2['trainset']
    X_spec = np.concatenate([X1_spec, X2_spec])
    X_ecg = np.concatenate([X1_ecg, X2_ecg])
    y = np.concatenate([y1, y2])
    y = y[..., 0:3]
    # X = []
    # # Y = []
    # for i in range(len(X_ecg)):
    #     ecg = np.array(X_ecg[i])
    #     r_index = np.where(ecg==1)
    #     r_index = np.divide(r_index[0], 18176)
    #     X.append(r_index)
    #     diff = 300
    #     start = 0
    #     while start < 18176 - diff:
    #         ecg2save = np.zeros((18176))
    #         # print(len(ecg2save[0:(18176-start-diff)]))
    #         # print(len(ecg[(start+diff):18176]))
    #         ecg2save[0:(18176-start-diff)] = ecg[(start+diff):18176]
    #         ecg2save[(18176-start-diff):18176] = ecg[0:(start+diff)]
    #         X.append(ecg2save)
    #         Y.append(y[i])
    #         start += diff
    # X = np.array(X)
    # Y = np.array(Y)
    # ind_train = np.random.permutation(len(X))
    # X = X[ind_train]
    # Y = Y[ind_train]

    load_fn5 = '/data/weiyuhua/data/AF_preprocessed_data/data_aug_test.mat'
    load_data5 = scipy.io.loadmat(load_fn5)
    final_testset_ecg = load_data5['final_testset']
    final_testset_spec = load_data5['testsetSpec']
    final_testtarget = load_data5['final_testtarget']
    final_testtarget = final_testtarget[..., 0:3]

    load_fn_val = '/data/weiyuhua/data/AF_preprocessed_data/data_aug_val.mat'
    load_data_val = scipy.io.loadmat(load_fn_val)
    val_set_ecg = load_data_val['val_set']
    val_set_spec = load_data_val['valsetSpec']
    val_target = load_data_val['val_target']
    val_target = val_target[..., 0:3]

    load_fn_r = '/data/weiyuhua/data/AF_preprocessed_data/data_aug-rwave.mat'
    load_data_r = scipy.io.loadmat(load_fn_r)
    R = load_data_r['trainsetPeaks-r']
    val_R = load_data_r['valsetPeaks-r']
    test_R = load_data_r['testsetPeaks-r']
    R_index = []
    val_R_index = []
    test_R_index = []
    # Y = []

    for i in range(len(R)):
        recg = np.array(R[i])
        r_index = np.where(recg==1)
        # r_index = np.divide(r_index[0], 18176)
        R_index.append(r_index[0])

    true_box_buffer = 200
    box_number = 1

    R_labels = np.zeros((len(R_index), 1136, box_number, 2))
    R_box = np.zeros((len(R_index), 1,1,true_box_buffer,1))
    true_box_index = 0
    for i in range(len(R_index)):
        old_grid = -1
        count = 0
        for r in R_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            # count = int(np.floor((scaled_index - grid) * box_number))
            box = [scaled_index]
            R_labels[i, grid, count, 0:1] = box
            R_labels[i, grid, count, 1] = 1.
            R_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer
            # count += 1
            # if (old_grid != grid):
            #     count = 0
            #     old_grid = grid

    for i in range(len(val_R)):
        recg = np.array(val_R[i])
        r_index = np.where(recg==1)
        # r_index = np.divide(r_index[0], 18176)
        val_R_index.append(r_index[0])
    val_R_labels = np.zeros((len(val_R_index), 1136, box_number, 2))
    val_R_box = np.zeros((len(val_R_index), 1, 1, true_box_buffer, 1))
    true_box_index = 0
    for i in range(len(val_R_index)):
        count = 0
        # old_grid = -1
        for r in val_R_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            # count = int(np.floor((scaled_index - grid) * box_number))
            box = [scaled_index]
            val_R_labels[i, grid, count, 0:1] = box
            val_R_labels[i, grid, count, 1] = 1.
            val_R_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer
            # count += 1
            # if (old_grid != grid):
            #     count = 0
            #     old_grid = grid


    for i in range(len(test_R)):
        recg = np.array(test_R[i])
        r_index = np.where(recg==1)
        # r_index = np.divide(r_index[0], 18176)
        test_R_index.append(r_index[0])
    test_R_labels = np.zeros((len(test_R_index), 1136, box_number, 2))
    test_R_box = np.zeros((len(test_R_index), 1, 1, true_box_buffer, 1))

    true_box_index = 0
    for i in range(len(test_R_index)):
        count = 0
        # old_grid = -1
        for r in test_R_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            # count = int(np.floor((scaled_index - grid) * box_number))
            box = [scaled_index]
            test_R_labels[i, grid, count, 0:1] = box
            test_R_labels[i, grid, count, 1] = 1.
            test_R_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer
            # count += 1
            # if (old_grid != grid):
            #     count = 0
            #     old_grid = grid


    load_fn_p = '/data/weiyuhua/data/AF_preprocessed_data/data_aug-pwave.mat'
    load_data_p = scipy.io.loadmat(load_fn_p)
    P = load_data_p['trainsetPeaks-p']
    val_P = load_data_p['valsetPeaks-p']
    test_P = load_data_p['testsetPeaks-p']
    P_index = []
    val_P_index = []
    test_P_index = []

    for i in range(len(P)):
        recg = np.array(P[i])
        p_index = np.where(recg==1)
        P_index.append(p_index[0])

    true_box_buffer = 200
    box_number = 1

    P_labels = np.zeros((len(P_index), 1136, box_number, 2))
    P_box = np.zeros((len(P_index), 1,1,true_box_buffer,1))
    true_box_index = 0
    for i in range(len(P_index)):
        count = 0
        for r in P_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            P_labels[i, grid, count, 0:1] = box
            P_labels[i, grid, count, 1] = 1.
            P_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer

    for i in range(len(val_P)):
        recg = np.array(val_P[i])
        p_index = np.where(recg==1)
        val_P_index.append(p_index[0])
    val_P_labels = np.zeros((len(val_P_index), 1136, box_number, 2))
    val_P_box = np.zeros((len(val_P_index), 1, 1, true_box_buffer, 1))
    true_box_index = 0
    for i in range(len(val_P_index)):
        count = 0
        # old_grid = -1
        for r in val_P_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            val_P_labels[i, grid, count, 0:1] = box
            val_P_labels[i, grid, count, 1] = 1.
            val_P_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer


    for i in range(len(test_P)):
        recg = np.array(test_P[i])
        p_index = np.where(recg==1)
        test_P_index.append(p_index[0])
    test_P_labels = np.zeros((len(test_P_index), 1136, box_number, 2))
    test_P_box = np.zeros((len(test_P_index), 1, 1, true_box_buffer, 1))

    true_box_index = 0
    for i in range(len(test_P_index)):
        count = 0
        for r in test_P_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            test_P_labels[i, grid, count, 0:1] = box
            test_P_labels[i, grid, count, 1] = 1.
            test_P_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer

    load_fn_q = '/data/weiyuhua/data/AF_preprocessed_data/data_aug-qwave.mat'
    load_data_q = scipy.io.loadmat(load_fn_q)
    Q = load_data_q['trainsetPeaks-q']
    val_Q = load_data_q['valsetPeaks-q']
    test_Q = load_data_q['testsetPeaks-q']
    Q_index = []
    val_Q_index = []
    test_Q_index = []

    for i in range(len(Q)):
        recg = np.array(Q[i])
        p_index = np.where(recg == 1)
        Q_index.append(p_index[0])

    true_box_buffer = 200
    box_number = 1

    Q_labels = np.zeros((len(Q_index), 1136, box_number, 2))
    Q_box = np.zeros((len(Q_index), 1, 1, true_box_buffer, 1))
    true_box_index = 0
    for i in range(len(Q_index)):
        count = 0
        for r in Q_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            Q_labels[i, grid, count, 0:1] = box
            Q_labels[i, grid, count, 1] = 1.
            Q_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer

    for i in range(len(val_Q)):
        recg = np.array(val_Q[i])
        p_index = np.where(recg == 1)
        val_Q_index.append(p_index[0])
    val_Q_labels = np.zeros((len(val_Q_index), 1136, box_number, 2))
    val_Q_box = np.zeros((len(val_Q_index), 1, 1, true_box_buffer, 1))
    true_box_index = 0
    for i in range(len(val_Q_index)):
        count = 0
        # old_grid = -1
        for r in val_Q_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            val_Q_labels[i, grid, count, 0:1] = box
            val_Q_labels[i, grid, count, 1] = 1.
            val_Q_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer

    for i in range(len(test_Q)):
        recg = np.array(test_Q[i])
        p_index = np.where(recg == 1)
        test_Q_index.append(p_index[0])
    test_Q_labels = np.zeros((len(test_Q_index), 1136, box_number, 2))
    test_Q_box = np.zeros((len(test_Q_index), 1, 1, true_box_buffer, 1))

    true_box_index = 0
    for i in range(len(test_Q_index)):
        count = 0
        for r in test_Q_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            test_Q_labels[i, grid, count, 0:1] = box
            test_Q_labels[i, grid, count, 1] = 1.
            test_Q_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer

    load_fn_t = '/data/weiyuhua/data/AF_preprocessed_data/data_aug-twave.mat'
    load_data_t = scipy.io.loadmat(load_fn_t)
    T = load_data_t['trainsetPeaks-t']
    val_T = load_data_t['valsetPeaks-t']
    test_T = load_data_t['testsetPeaks-t']
    T_index = []
    val_T_index = []
    test_T_index = []

    for i in range(len(T)):
        recg = np.array(T[i])
        p_index = np.where(recg == 1)
        T_index.append(p_index[0])

    true_box_buffer = 200
    box_number = 1

    T_labels = np.zeros((len(T_index), 1136, box_number, 2))
    T_box = np.zeros((len(T_index), 1, 1, true_box_buffer, 1))
    true_box_index = 0
    for i in range(len(T_index)):
        count = 0
        for r in T_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            T_labels[i, grid, count, 0:1] = box
            T_labels[i, grid, count, 1] = 1.
            T_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer

    for i in range(len(val_T)):
        recg = np.array(val_T[i])
        p_index = np.where(recg == 1)
        val_T_index.append(p_index[0])
    val_T_labels = np.zeros((len(val_T_index), 1136, box_number, 2))
    val_T_box = np.zeros((len(val_T_index), 1, 1, true_box_buffer, 1))
    true_box_index = 0
    for i in range(len(val_T_index)):
        count = 0
        # old_grid = -1
        for r in val_T_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            val_T_labels[i, grid, count, 0:1] = box
            val_T_labels[i, grid, count, 1] = 1.
            val_T_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer

    for i in range(len(test_T)):
        recg = np.array(test_T[i])
        p_index = np.where(recg == 1)
        test_T_index.append(p_index[0])
    test_T_labels = np.zeros((len(test_T_index), 1136, box_number, 2))
    test_T_box = np.zeros((len(test_T_index), 1, 1, true_box_buffer, 1))

    true_box_index = 0
    for i in range(len(test_T_index)):
        count = 0
        for r in test_T_index[i]:
            scaled_index = r / (18176 / 1136)
            grid = int(np.floor(scaled_index))
            box = [scaled_index]
            test_T_labels[i, grid, count, 0:1] = box
            test_T_labels[i, grid, count, 1] = 1.
            test_T_box[i, 0, 0, true_box_index] = box
            true_box_index += 1
            true_box_index = true_box_index % true_box_buffer

    # Merging datasets
    # Case other sets are available, load them then concatenate
    # y = np.concatenate((traintarget,augtarget),axis=0)
    # X = np.concatenate((trainset,augset),axis=0)

    # X =  X[:,0:WINDOW_SIZE]
    return ([X_ecg, X_spec], y), ([val_set_ecg, val_set_spec], val_target), ([final_testset_ecg, final_testset_spec], final_testtarget),\
           ([R_labels,R_box], [val_R_labels, val_R_box], [test_R_labels, test_R_box]), \
           ([P_labels, P_box], [val_P_labels, val_P_box], [test_P_labels, test_P_box]), \
           ([Q_labels, Q_box], [val_Q_labels, val_Q_box], [test_Q_labels, test_Q_box]), \
           ([T_labels, T_box], [val_T_labels, val_T_box], [test_T_labels, test_T_box])
    # return (X[0:100], y[0:100]),(val_set[0:100], val_target[0:100])