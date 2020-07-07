import tensorflow.keras as keras
from  tensorflow.keras import backend as K
import numpy as np
import time

class Classifier_INCEPTION:

    def __init__(self, input_shape, nb_classes, type,   # 默认64
                 nb_filters=16, use_residual=True, use_bottleneck=True, bottleneck_size=16, depth=6, kernel_size=81,use_attention=False,
                 attention_module='SE',se_ratio=16,cbam_ratio=16, head_dropout_rate=0):

        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = bottleneck_size
        self.use_attention = use_attention
        self.attention_module = attention_module
        self.se_ratio = se_ratio
        self.cbam_ratio = cbam_ratio
        self.head_dropout_rate = head_dropout_rate
        if type=="inceptiontime":
            self.model = self.build_model(input_shape, nb_classes)
        elif type=="inceptiontime_v2":
            self.model = self.build_model_v2(input_shape, nb_classes)

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        # [40,20,10]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)

        return x

    def _shortcut_layer(self, input_tensor, out_tensor, stride=1):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same',strides=stride, use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def _attention_module(self,input_feature):
        if self.attention_module == 'SE':  # SE_block
            x = self._squeeze_excitation_block(input_feature)
        elif self.attention_module == 'CBAM':  # CBAM_block
            x = self._cbam_block(input_feature)
        else:
            raise Exception("'{}' is not supported attention module!".format(self.attention_module))
        return x

    def _squeeze_excitation_block(self, input_feature):
        '''
        SE module performs inter-channel weighting.

         """Contains the implementation of Squeeze-and-Excitation(SE) block.
        As described in https://arxiv.org/abs/1709.01507.
        """

        '''
        channel_axis = -1
        channel = input_feature.shape[channel_axis]
        # print(channel)

        squeeze = keras.layers.GlobalAveragePooling1D()(input_feature)

        excitation = keras.layers.Dense(units=channel // self.se_ratio)(squeeze)
        excitation = keras.layers.Activation(activation='relu')(excitation)
        excitation = keras.layers.Dense(units=channel)(excitation)
        excitation = keras.layers.Activation(activation='sigmoid')(excitation)
        excitation = keras.layers.Reshape((1,channel))(excitation)

        scale = keras.layers.Multiply()([input_feature, excitation])
        return scale

    def _cbam_block(self, input_feature, channel=True, spatial=True, mode = 'channel-first'):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
           As described in https://arxiv.org/abs/1807.06521.
           """
        if channel & spatial:
            if mode == 'channel-first':
                cbam_feature = self._cbam_channel_attention(input_feature)
                cbam_feature = self._cbam_spatial_attention(cbam_feature)
            else:
                cbam_feature = self._cbam_spatial_attention(input_feature)
                cbam_feature = self._cbam_channel_attention(cbam_feature)
        elif channel:
            cbam_feature = self._cbam_channel_attention(input_feature)
        elif spatial:
            cbam_feature = self._cbam_spatial_attention(input_feature)
        else:
            raise Exception("at least one attention module!")
        return cbam_feature

    def _cbam_channel_attention(self, input_feature):
        channel_axis = -1
        channel = input_feature.shape[channel_axis]

        shared_layer_one = keras.layers.Dense(channel // self.cbam_ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')
        shared_layer_two = keras.layers.Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        avg_pool = keras.layers.GlobalAveragePooling1D()(input_feature)
        avg_pool = keras.layers.Reshape((1, channel))(avg_pool)
        assert avg_pool.shape[1:] == (1, channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1, channel // self.cbam_ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1, channel)

        max_pool = keras.layers.GlobalMaxPooling1D()(input_feature)
        max_pool = keras.layers.Reshape((1, channel))(max_pool)
        assert max_pool.shape[1:] == (1, channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1, channel // self.cbam_ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1, channel)

        cbam_feature = keras.layers.Add()([avg_pool, max_pool])
        cbam_feature = keras.layers.Activation('sigmoid')(cbam_feature)

        return keras.layers.multiply([input_feature, cbam_feature])

    def _cbam_spatial_attention(self, input_feature):
        kernel_size = 7

        channel = input_feature.shape[-1]
        cbam_feature = input_feature

        avg_pool = keras.layers.Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = keras.layers.Lambda(lambda x: K.max(x, axis=2, keepdims=True))(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = keras.layers.Concatenate(axis=2)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = keras.layers.Conv1D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        assert cbam_feature.shape[-1] == 1

        return keras.layers.multiply([input_feature, cbam_feature])

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_attention:
                x = self._attention_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.summary()

        return model

    def build_model_v2(self,input_shape, nb_classes):

        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            if self.use_attention:
                x = self._inception_module(x, 2)
                x = self._attention_module(x)
                x = self._inception_module(x, 1)
                x = self._attention_module(x)
            else:
                x = self._inception_module(x, 2)
                x = self._inception_module(x, 1)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, 8)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        if self.head_dropout_rate > 0:
            x = keras.layers.Dropout(self.head_dropout_rate)(gap_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.summary()

        return model


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = Classifier_INCEPTION((18176,1), 5, type='inceptiontime_v2',use_attention=True,
                 attention_module='CBAM').model
    model.summary()