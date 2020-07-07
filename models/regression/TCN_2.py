import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

class block(keras.Model):
    def __init__(self, input_shape, filters, kernel_size, stride, dilation, dropout):
        super(block, self).__init__()
        # 此处忽略了weightNormalization层
        self.conv1 = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='causal',
                dilation_rate=dilation,
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
            )
        self.dropout1 = keras.layers.Dropout(dropout)
        self.conv2 = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='causal',
                dilation_rate=dilation,
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
            )
        self.dropout2 = keras.layers.Dropout(dropout)
        self.res = keras.layers.Conv1D(filters=filters, kernel_size=1)
        self.relu = keras.layers.Activation('relu')

    def call(self, input):
        x = self.conv1(input)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        y = self.res(input)
        z = x + y
        z = self.relu(z)
        return z

class TCN2(keras.Model):
    def __init__(self, input_shape, output_shape):
        super(TCN2, self).__init__()
        self.stride = 1
        self.dropout = 0.2
        self.filters = [32, 32]
        self.num_levels = 2
        self.kernel_size = 3
        self.block1 = block(input_shape, 32, self.kernel_size, self.stride, 1, self.dropout)
        self.block2 = block([input_shape[0], 32], 32, self.kernel_size, self.stride, 2, self.dropout)
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(output_shape, activation='sigmoid')


    def call(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
