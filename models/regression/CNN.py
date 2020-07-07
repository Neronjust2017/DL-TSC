import tensorflow.keras as keras

class CNN(keras.Model):
    def __init__(self, input_shape, output_shape):
        super(CNN, self).__init__()

        self.k_num = 32
        self.k_size = 2
        self.k_str = 2

        self.conv1 = keras.layers.Conv1D(filters=self.k_num,
               kernel_size=self.k_size,
               padding='same',
               strides=self.k_str)
        self.conv2 = keras.layers.Conv1D(filters=self.k_num,
               kernel_size=self.k_size,
               padding='same',
               strides=self.k_str)
        self.conv3 = keras.layers.Conv1D(filters=self.k_num * 2,
               kernel_size=self.k_size,
               padding='same',
               strides=self.k_str)
        self.conv4 = keras.layers.Conv1D(filters=self.k_num * 2,
               kernel_size=self.k_size,
               padding='same',
               strides=self.k_str)
        self.flatten = keras.layers.Flatten()
        self.relu = keras.layers.Activation('relu')
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.batch_norm3 = keras.layers.BatchNormalization()
        self.batch_norm4 = keras.layers.BatchNormalization()
        self.max_pool1 = keras.layers.MaxPooling1D(pool_size=2,
                      strides=1, padding='same')
        self.max_pool2 = keras.layers.MaxPooling1D(pool_size=2,
                      strides=1, padding='same')
        self.dense = keras.layers.Dense(output_shape, activation='sigmoid')
    def call(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.max_pool1(x)

        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.max_pool2(x)

        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.flatten(x)

        x = self.dense(x)
        return x

# if __name__ == '__main__':
#     x = np.zeros((32, 168, 15))
#     y = np.zeros((32, 15))
#     model = Model()
#     model.compile(loss='mse', optimizer='adam')
#     model.fit(x, y, epochs=2)
#     print('Done')
