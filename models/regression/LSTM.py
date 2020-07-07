# 这里需要使用tensorflow.keras，否则可能会出现一个奇怪的错误：
# AttributeError: 'TFOptimizer' object has no attribute 'learning_rate'
import tensorflow.keras as keras

class LSTM:
    def __init__(self, input_shape, output_shape):
        self.model = self.build_model(input_shape, output_shape)

    def build_model(self, input_shape, output_shape):
        '''
                1. 理解LSTM的输出，以及参数return_sequences 和 return_state
                    https://huhuhang.com/post/machine-learning/lstm-return-sequences-state
                    keras中默认return_sequences=False 和 return_state=False
                    默认状态下输出为最后时刻的隐藏状态ht, 格式为(samples, output_dim)
                    如果return_sequences=True, 则输出每一个时刻的隐藏状态h, 格式为(samples, timesteps, output_dim)
                    如果return_state=True, 则输出3个值，0和1都是最后时刻的隐藏状态ht，2是最后时刻的细胞状态ct
                    如果二者都为True，则输出3个值：
                    0 - 所有时刻的隐藏状态h
                    1 - 最后时刻的隐藏状态ht
                    2 - 最后时刻的细胞状态ct
                    如果是多层LSTM，需要将前N-1层return_sequence=True, 最后一层为False
                2. dropout 与 recurrent_dropout
                    https://segmentfault.com/a/1190000017318397

                3. batchNormal怎么用
                    “BN层放在每一个全连接层和激励函数之间”
                    LSTM可以使用该层，但GRU有时会出现loss为nan

        '''



        input_layer = keras.layers.Input(shape=(input_shape[0], input_shape[1]))
        lstm1 = keras.layers.LSTM(32,
                                  return_sequences=True,
                                  dropout=0.4,
                                  recurrent_dropout=0.4)(input_layer)

        bn1 = keras.layers.BatchNormalization()(lstm1)
        relu1 = keras.layers.Activation('relu')(bn1)
        lstm2 = keras.layers.LSTM(64,
                                  dropout=0.4,
                                  recurrent_dropout=0.4,
                                  return_state=True)(relu1)
        lstm2 = lstm2[1]
        bn2 = keras.layers.BatchNormalization()(lstm2)
        relu2 = keras.layers.Activation('relu')(bn2)
        #dense = keras.layers.Dense(output_shape, activation='sigmoid')(relu2)
        dense = keras.layers.Dense(output_shape, activation='softmax')(relu2)

        model = keras.models.Model(inputs=input_layer, outputs=dense)

        return model


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model =LSTM(input_shape=(18000,12), output_shape=9).model
    model.summary()