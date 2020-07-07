# -*- coding:utf-8 -*-
from base.base_model import BaseModel
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, BatchNormalization, Activation, Dense
import tensorflow.keras.backend as K
from models.regression.LSTM import LSTM
from models.regression.DeepConvLSTM import DeepConvLSTM
from models.regression.DeepConvLSTM_2 import DeepConvLSTM2
from models.regression.DeepResBiLSTM import DeepResBiLSTM
from models.regression.TCN_3 import TemporalConvNet
from models.regression.TCN import TCN
from models.regression.TCN_2 import TCN2
from models.regression.CNN import CNN


def root_mean_square_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))




def explained_variance_score(y_true, y_pred):
    a_mean = K.mean(y_true - y_pred)
    a = K.mean(K.square(y_true - y_pred - a_mean))
    b_mean = K.mean(y_true)
    b = K.mean(K.square(y_true - b_mean))
    return 1 - a / b


class UtsRegressionModel(BaseModel):
    def __init__(self, config, input_shape, output_shape):
        super(UtsRegressionModel, self).__init__(config)
        self.build_model(input_shape, output_shape)

    def build_model(self, input_shape, output_shape):
        '''
        self.model = Sequential()
        self.model.add(keras.layers.Input(shape=(input_shape[0], input_shape[1])))
        self.model.add(keras.layers.LSTM(32))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dense(output_shape, activation='sigmoid'))
        '''
        if self.config.args.model == 'LSTM':
            self.model = LSTM(input_shape, output_shape).model
        if self.config.args.model == 'DeepConvLSTM':
            # 对应DeepConvLSTM-1模型写法
            self.model = DeepConvLSTM(input_shape, output_shape).model

            # 对应DeepConvLSTM-2模型写法，如果不加第二句话会报错，提示模型没有build
            #self.model = DeepConvLSTM2(input_shape, output_shape)
            #self.model(tf.zeros((input_shape[2], input_shape[0], input_shape[1])))

        if self.config.args.model == 'DeepResBiLSTM':
            self.model = DeepResBiLSTM(input_shape, output_shape).model

        if self.config.args.model == 'TCN':
            # 对应TCN-1模型
            #self.model = TCN(input_shape, output_shape).model

            # 对应TCN-2模型
            # 第2句和第3句保留一句即可，如果不加，会报错
            self.model = TCN2(input_shape, output_shape)
            self.model.build((input_shape[2], input_shape[0], input_shape[1]))
            #self.model(tf.zeros((input_shape[2], input_shape[0], input_shape[1])))

        if self.config.args.model == 'CNN':
            self.model = CNN(input_shape, output_shape)
            self.model.build((input_shape[2], input_shape[0], input_shape[1]))

        self.model.compile(
            loss='mse',
            optimizer=self.config.args.optimizer,
            metrics=['mae', 'mse', root_mean_square_error, 'mape', 'msle', explained_variance_score]
        )
        self.model.summary()