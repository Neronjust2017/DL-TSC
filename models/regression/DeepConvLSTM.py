# 函数式API模型写法参考资料
# https://keras.io/zh/getting-started/functional-api-guide/
#
#

# 尽量避免 keras 和 tensorflow 混用，建议使用tensorflow.keras
#import keras
import tensorflow as tf
from tensorflow import keras

class DeepConvLSTM:
    def __init__(self, input_shape, output_shape):
        self.model = self.build_model(input_shape, output_shape)

    def build_model(self, input_shape, output_shape):

        # 注意这里不要把输入层命名为input，否则会报一些奇奇怪怪的错误
        # 注意输入请单独起一个名称且下文中不要再赋值，否则之后会报错
        # 注意如果下文有reshape操作并且使用了-1的自动推断，这里应该使用batch_shape，否则下面会报错
        # 注意这里对张量的操作都是用层来做，例如两个张量求和：keras.layers.add([x, y])，两个张量拼接：keras.layers.concatenate([out_x, out_y])
        input_layer = keras.layers.Input(batch_shape=(input_shape[2], input_shape[0], input_shape[1]))
        #input_layer = keras.layers.Input(shape=(input_shape[0], input_shape[1]))

        # 输入为（128， 168， 321），需要把它调整为（128， 168， 1， 321）
        # 两种写法都可以，推荐使用第二种，注意第二种方法写的时候是尺寸是不包括 batch_size 的
        #input_reshape = keras.layers.Lambda(lambda x: tf.reshape(x, [input_shape[2], input_shape[0], 1, input_shape[1]]))(input_layer)
        input_reshape = keras.layers.Reshape((input_shape[0], 1, input_shape[1]))(input_layer)

        conv1 = keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            padding='same',
            strides=1
        )(input_reshape)
        relu = keras.layers.Activation('relu')(conv1)
        conv2 = keras.layers.Conv2D(
            filters=128,
            kernel_size=5,
            padding='same',
            strides=1
        )(relu)
        relu = keras.layers.Activation('relu')(conv2)
        conv3 = keras.layers.Conv2D(
            filters=256,
            kernel_size=5,
            padding='same',
            strides=1
        )(relu)
        relu = keras.layers.Activation('relu')(conv3)
        conv4 = keras.layers.Conv2D(
            filters=512,
            kernel_size=5,
            padding='same',
            strides=1
        )(relu)
        relu = keras.layers.Activation('relu')(conv4)

        # 这里对输入LSTM的张量维度进行调整
        #relu = keras.layers.Lambda(lambda x: tf.reshape(x, [input_shape[2], input_shape[0], -1]))(relu)
        relu = keras.layers.Reshape((input_shape[0], -1))(relu)
        lstm1 = keras.layers.LSTM(128,
                                  return_sequences=True,
                                  dropout=0.5,
                                  recurrent_dropout=0.5)(relu)
        relu = keras.layers.Activation('relu')(lstm1)
        lstm2 = keras.layers.LSTM(128,
                                  dropout=0.5,
                                  recurrent_dropout=0.5)(relu)
        relu = keras.layers.Activation('relu')(lstm2)
        #output_layer = keras.layers.Dense(output_shape, activation='sigmoid')(relu)
        output_layer = keras.layers.Dense(output_shape, activation='softmax')(relu)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model =DeepConvLSTM(input_shape=(128,18000,12), output_shape=9).model
    model.summary()