import tensorflow.keras as keras
import tensorflow as tf

class DeepConvLSTM2(keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DeepConvLSTM2, self).__init__()
        self.samples = input_dim[2]
        self.channels = input_dim[0]
        self.rows = 1
        self.cols = input_dim[1]

        self.conv1 = keras.layers.Conv2D(
            input_shape=(self.channels, self.rows, self.cols),
            filters=64,
            kernel_size=5,
            padding='same',
            strides=1
        )
        self.conv2 = keras.layers.Conv2D(
            filters=128,
            kernel_size=5,
            padding='same',
            strides=1
        )
        self.conv3 = keras.layers.Conv2D(
            filters=256,
            kernel_size=5,
            padding='same',
            strides=1
        )
        self.conv4 = keras.layers.Conv2D(
            filters=512,
            kernel_size=5,
            padding='same',
            strides=1
        )
        self.relu = keras.layers.Activation('relu')
        self.lstm1 = keras.layers.LSTM(128,
                                    return_sequences = True,
                                    dropout = 0.5,
                                    recurrent_dropout = 0.5)
        self.lstm2 = keras.layers.LSTM(128,
                                       dropout=0.5,
                                       recurrent_dropout=0.5)
        self.dense = keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, x):
        #x.reshape(-1, self.channels, self.rows, self.cols)
        x = tf.reshape(x,[self.samples, self.channels, self.rows, self.cols])
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = tf.reshape(x, [self.samples, self.channels, -1])
        x = self.lstm1(x)
        x = self.relu(x)
        x = self.lstm2(x)
        x = self.relu(x)
        x = self.dense(x)
        return x
