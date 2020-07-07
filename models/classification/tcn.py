import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import numpy as np
from keras.utils.vis_utils import plot_model

class Classifier_TemporalConvNet:
    def __init__(self, input_shape, nb_classes, stride = 1, dropout = 0.5, filters=None, num_levels = 3 ,kernel_size = 64):

        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.stride = stride
        self.dropout = dropout

        if filters is None:
            self.filters = [64, 64, 64, 64, 64, 64, 64, 64]
        else:
            self.filters = filters

        self.num_levels = num_levels
        self.kernel_size = kernel_size

        self.model = self._build_model(input_shape, nb_classes)


    def _build_model(self, input_shape, nb_classes):
        global block
        input_layer = keras.layers.Input(input_shape)
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            filters = self.filters[i]
            if i==0:
                block = self._build_block(input_layer, filters, self.kernel_size, self.stride, dilation_size, self.dropout)
            else:
                block = self._build_block(block, filters, self.kernel_size, self.stride, dilation_size, self.dropout)
        x = keras.layers.Lambda(lambda x: x[:,-1,:])(block)
        # x = keras.layers.Flatten()(block)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(x)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.summary()
        return model

    def _build_block(self, input_tensor, filters, kernel_size, stride, dilation, dropout):

        conv1 = tfa.layers.WeightNormalization(keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='causal',
            dilation_rate=dilation,
            activation='relu',
            kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
        ))(input_tensor)

        dropout1 = keras.layers.Dropout(dropout)(conv1)

        conv2 = tfa.layers.WeightNormalization(
            keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='causal',
            dilation_rate=dilation,
            activation='relu',
            kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
        ))(dropout1)

        out = keras.layers.Dropout(dropout)(conv2)
        res = keras.layers.Conv1D(filters=filters, kernel_size=1)(input_tensor)
        out2 = keras.layers.add([out, res])
        output_layer = keras.layers.Activation('relu')(out2)
        return output_layer

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = Classifier_TemporalConvNet((18176,1), 5).model
    model.summary()