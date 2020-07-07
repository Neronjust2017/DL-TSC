# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Multiply, GlobalAveragePooling1D, Add, Dense, Activation, ZeroPadding1D, \
    BatchNormalization, Flatten, Conv1D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, Lambda, UpSampling1D, Reshape
from tensorflow.keras.models import Model, load_model
from keras.initializers import glorot_uniform
import keras.backend as K
from keras.callbacks import TensorBoard
from utils.AFClassication.data import loaddata
from utils.uts_classification.tools import AdvancedLearnignRateScheduler
from utils.uts_classification.metric import f1,recall,precision



def res_conv(X, filters, base, s):
    name_base = base + '/branch'

    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####

    X_shortcut = X

    ##### Branch1 #####
    # First component of Branch1
    X = BatchNormalization(name=name_base + '1/bn_1')(X)
    X = Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv1D(filters=F1, kernel_size=16, strides=1, padding='same', name=name_base + '1/conv_1',
               kernel_initializer=glorot_uniform(seed=0))(X)

    # Second component of Branch1
    X = BatchNormalization(name=name_base + '1/bn_2')(X)
    X = Activation('relu', name=name_base + '1/relu_2')(X)
    X = Conv1D(filters=F2, kernel_size=48, strides=s, padding='same', name=name_base + '1/conv_2',
               kernel_initializer=glorot_uniform(seed=0))(X)

    # Third component of Branch1
    X = BatchNormalization(name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv1D(filters=F3, kernel_size=16, strides=1, padding='same', name=name_base + '1/conv_3',
               kernel_initializer=glorot_uniform(seed=0))(X)

    ##### Branch2 ####
    X_shortcut = BatchNormalization(name=name_base + '2/bn_1')(X_shortcut)
    X_shortcut = Activation('relu', name=name_base + '2/relu_1')(X_shortcut)
    X_shortcut = Conv1D(filters=F3, kernel_size=16, strides=s, padding='same', name=name_base + '2/conv_1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    # Final step: Add Branch1 and Branch2
    X = Add(name=base + '/Add')([X, X_shortcut])

    return X


def res_identity(X, filters, base):
    name_base = base + '/branch'

    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####

    X_shortcut = X

    ##### Branch1 #####
    # First component of Branch1
    X = BatchNormalization(name=name_base + '1/bn_1')(X)
    Shortcut = Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv1D(filters=F1, kernel_size=16, strides=1, padding='same', name=name_base + '1/conv_1',
               kernel_initializer=glorot_uniform(seed=0))(Shortcut)

    # Second component of Branch1
    X = BatchNormalization(name=name_base + '1/bn_2')(X)
    X = Activation('relu', name=name_base + '1/relu_2')(X)
    X = Conv1D(filters=F2, kernel_size=48, strides=1, padding='same', name=name_base + '1/conv_2',
               kernel_initializer=glorot_uniform(seed=0))(X)

    # Third component of Branch1
    X = BatchNormalization(name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv1D(filters=F3, kernel_size=16, strides=1, padding='same', name=name_base + '1/conv_3',
               kernel_initializer=glorot_uniform(seed=0))(X)

    # Final step: Add Branch1 and the original Input itself
    X = Add(name=base + '/Add')([X, X_shortcut])

    return X


def Trunk_block(X, F, base):
    name_base = base

    X = res_identity(X, F, name_base + '/Residual_id_1')
    X = res_identity(X, F, name_base + '/Residual_id_2')

    return X


def interpolation(input_tensor, ref_tensor, name):  # resizes input_tensor wrt. ref_tensor

    # resize_nearest_neighbor

    # L = ref_tensor.get_shape()[1]
    # print(input_tensor.shape)
    # x = Reshape((input_tensor.shape[1],1,input_tensor.shape[2]))(input_tensor)
    # print(x.shape)
    # x =  tf.compat.v1.image.resize_nearest_neighbor(x, [L, 1], name=name)
    # out = Reshape((x.shape[1],x.shape[3]))(x)
    # print(out.shape)

    # print(input_tensor.shape)
    out = UpSampling1D(size=ref_tensor.shape[1]//input_tensor.shape[1])(input_tensor)
    # print(out.shape)

    return out


def Attention_1(X, filters, base):
    F1, F2, F3 = filters

    name_base = base

    X = res_identity(X, filters, name_base + '/Pre_Residual_id')

    X_Trunk = Trunk_block(X, filters, name_base + '/Trunk')
    print("X_Trunk")
    print(X_Trunk.shape)
    X = MaxPooling1D(3, strides=2, padding='same', name=name_base + '/Mask/pool_3')(X)

    print(X.shape)
    X = res_identity(X, filters, name_base + '/Mask/Residual_id_3_Down')

    Residual_id_3_Down_shortcut = X

    Residual_id_3_Down_branched = res_identity(X, filters, name_base + '/Mask/Residual_id_3_Down_branched')

    X = MaxPooling1D(3,strides=2, padding='same', name=name_base + '/Mask/pool_2')(X)

    print(X.shape)
    X = res_identity(X, filters, name_base + '/Mask/Residual_id_2_Down')

    Residual_id_2_Down_shortcut = X

    Residual_id_2_Down_branched = res_identity(X, filters, name_base + '/Mask/Residual_id_2_Down_branched')

    X = MaxPooling1D(3, strides=2, padding='same', name=name_base + '/Mask/pool_1')(X)

    print(X.shape)
    X = res_identity(X, filters, name_base + '/Mask/Residual_id_1_Down')

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_1_Up')

    temp_name1 = name_base + "/Mask/Interpool_1"

    # X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_2_Down_shortcut, 'name': temp_name1})(X)
    X = UpSampling1D(size=Residual_id_2_Down_shortcut.shape[1] // X.shape[1], name=temp_name1)(X)

    print(X.shape)
    X = Add(name=base + '/Mask/Add_after_Interpool_1')([X, Residual_id_2_Down_branched])

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_2_Up')

    temp_name2 = name_base + "/Mask/Interpool_2"

    # X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_3_Down_shortcut, 'name': temp_name2})(X)
    X = UpSampling1D(size=Residual_id_3_Down_shortcut.shape[1] // X.shape[1], name=temp_name2)(X)

    print(X.shape)
    X = Add(name=base + '/Mask/Add_after_Interpool_2')([X, Residual_id_3_Down_branched])

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_3_Up')

    temp_name3 = name_base + "/Mask/Interpool_3"

    # X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk, 'name': temp_name3})(X)
    X = UpSampling1D(size=X_Trunk.shape[1] // X.shape[1], name=temp_name3)(X)

    print(X.shape)
    X = BatchNormalization(name=name_base + '/Mask/Interpool_3/bn_1')(X)

    X = Activation('relu', name=name_base + '/Mask/Interpool_3/relu_1')(X)

    X = Conv1D(F3, kernel_size=1, strides=1, padding='same', name=name_base + '/Mask/Interpool_3/conv_1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    print(X.shape)
    X = BatchNormalization(name=name_base + '/Mask/Interpool_3/bn_2')(X)

    X = Activation('relu', name=name_base + '/Mask/Interpool_3/relu_2')(X)

    X = Conv1D(F3, kernel_size=1, strides=1, padding='same', name=name_base + '/Mask/Interpool_3/conv_2',
               kernel_initializer=glorot_uniform(seed=0))(X)
    print(X.shape)
    X = Activation('sigmoid', name=name_base + '/Mask/sigmoid')(X)

    X = Multiply(name=name_base + '/Mutiply')([X_Trunk, X])

    X = Add(name=name_base + '/Add')([X_Trunk, X])

    X = res_identity(X, filters, name_base + '/Post_Residual_id')

    return X


def Attention_2(X, filters, base):
    F1, F2, F3 = filters

    name_base = base

    X = res_identity(X, filters, name_base + '/Pre_Residual_id')

    X_Trunk = Trunk_block(X, filters, name_base + '/Trunk')

    X = MaxPooling1D(3, strides=2, padding='same', name=name_base + '/Mask/pool_2')(X)

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_2_Down')

    Residual_id_2_Down_shortcut = X

    Residual_id_2_Down_branched = res_identity(X, filters, name_base + '/Mask/Residual_id_2_Down_branched')

    X = MaxPooling1D(3, strides=2, padding='same', name=name_base + '/Mask/pool_1')(X)

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_1_Down')

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_1_Up')

    temp_name1 = name_base + "/Mask/Interpool_1"

    # X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_2_Down_shortcut, 'name': temp_name1})(X)
    X = UpSampling1D(size=Residual_id_2_Down_shortcut.shape[1] // X.shape[1], name=temp_name1)(X)

    X = Add(name=base + '/Mask/Add_after_Interpool_1')([X, Residual_id_2_Down_branched])

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_2_Up')

    temp_name2 = name_base + "/Mask/Interpool_2"

    # X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk, 'name': temp_name2})(X)
    X = UpSampling1D(size=X_Trunk.shape[1] // X.shape[1], name=temp_name2)(X)

    X = BatchNormalization(name=name_base + '/Mask/Interpool_2/bn_1')(X)

    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_1')(X)

    X = Conv1D(F3, kernel_size=1, strides=1, padding='same', name=name_base + '/Mask/Interpool_2/conv_1',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(name=name_base + '/Mask/Interpool_2/bn_2')(X)

    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_2')(X)

    X = Conv1D(F3, kernel_size=1, strides=1, padding='same', name=name_base + '/Mask/Interpool_2/conv_2',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = Activation('sigmoid', name=name_base + '/Mask/sigmoid')(X)

    X = Multiply(name=name_base + '/Mutiply')([X_Trunk, X])

    X = Add(name=name_base + '/Add')([X_Trunk, X])

    X = res_identity(X, filters, name_base + '/Post_Residual_id')

    return X


def Attention_3(X, filters, base):
    F1, F2, F3 = filters

    name_base = base

    X = res_identity(X, filters, name_base + '/Pre_Residual_id')

    X_Trunk = Trunk_block(X, filters, name_base + '/Trunk')

    X = MaxPooling1D(3, strides=2, padding='same', name=name_base + '/Mask/pool_1')(X)

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_1_Down')

    X = res_identity(X, filters, name_base + '/Mask/Residual_id_1_Up')

    temp_name2 = name_base + "/Mask/Interpool_1"

    # X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk, 'name': temp_name2})(X)
    X = UpSampling1D(size=X_Trunk.shape[1] // X.shape[1], name=temp_name2)(X)

    X = BatchNormalization(name=name_base + '/Mask/Interpool_2/bn_1')(X)

    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_1')(X)

    X = Conv1D(F3, kernel_size=1, strides=1, padding='same', name=name_base + '/Mask/Interpool_2/conv_1',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(name=name_base + '/Mask/Interpool_2/bn_2')(X)

    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_2')(X)

    X = Conv1D(F3, kernel_size=1, strides=1, padding='same', name=name_base + '/Mask/Interpool_2/conv_2',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = Activation('sigmoid', name=name_base + '/Mask/sigmoid')(X)

    X = Multiply(name=name_base + '/Mutiply')([X_Trunk, X])

    X = Add(name=name_base + '/Add')([X_Trunk, X])

    X = res_identity(X, filters, name_base + '/Post_Residual_id')

    return X



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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

X_input = Input(input_shape)

X = Conv1D(8, 7, strides=2, padding='same', name='conv_1', kernel_initializer=glorot_uniform(seed=0))(
    X_input)
X = BatchNormalization(axis=-1, name='bn_1')(X)
X = Activation('relu', name='relu_1')(X)
X = MaxPooling1D(3, strides=2, padding='same', name='pool_1')(X)
X = res_conv(X, [8, 8, 32], 'Residual_conv_1', 1)

### Attention 1 Start
X = Attention_1(X, [8, 8, 32], 'Attention_1')
### Attention 1 End

X = res_conv(X, [16, 16, 64], 'Residual_conv_2', 2)

### Attention 2 Start
X = Attention_2(X, [16, 16, 64], 'Attention_2')
### Attention 2 End

X = res_conv(X, [32, 32, 128], 'Residual_conv_3', 2)

### Attention 3 Start
X = Attention_3(X, [32, 32, 128], 'Attention_3')
### Attention 3 End

X = res_conv(X, [64, 64, 256], 'Residual_conv_4', 2)

X = res_identity(X, [64, 64, 256], 'Residual_id_1')
X = res_identity(X, [64, 64, 256], 'Residual_id_2')
X = BatchNormalization(name='bn_2')(X)
X = Activation('relu', name='relu_2')(X)
print(X.shape)

X = GlobalAveragePooling1D()(X)
print(X.shape)

X = Dense(nb_classes,activation='sigmoid', name='Dense_1')(X)
model = Model(inputs=X_input, outputs=X, name='attention_56')

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy',recall,precision,f1])
tbCallBack = TensorBoard(log_dir='tensorboard_log',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

model.fit(x_train[0:train_number],y_train[0:train_number], epochs=100, validation_split=0.1, batch_size=16, callbacks=[keras.callbacks.EarlyStopping(patience=10),
    AdvancedLearnignRateScheduler(monitor='val_loss', patience=6, verbose=1, mode='auto', decayRatio=0.1, warmup_batches=5, init_lr=0.001)],verbose=1)

loss, accuracy, recall, precision, f1 = model.evaluate(final_testset[0], final_testtarget)
print(loss)
print(accuracy)
print(recall)
print(precision)
print(f1)