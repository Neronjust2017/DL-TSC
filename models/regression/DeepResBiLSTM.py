import tensorflow as tf
import tensorflow.keras as keras

class DeepResBiLSTM:
    def __init__(self, input_shape, output_shape):
        self.dropout = 0.15
        self.nhide = 32
        self.binhide = 16
        self.model = self.build_model(input_shape, output_shape)
        pass

    def build_model(self, input_shape, output_shape):
        block_input = keras.layers.Input(batch_shape=(input_shape[2], input_shape[0], input_shape[1]))
        x = keras.layers.Dense(self.nhide, activation='relu')(block_input)
        #x = keras.layers.Activation('relu')(block_input)
        x = keras.layers.Bidirectional(keras.layers.LSTM(self.binhide))(x)
        y = keras.layers.Dense(self.nhide, activation='relu')(x)
        y = keras.layers.Reshape((1, y.shape[1]))(y)
        #y = keras.layers.Activation('relu')(x)
        y = keras.layers.Bidirectional(keras.layers.LSTM(self.binhide))(y)
        z = keras.layers.add([x, y])
        z = keras.layers.Dropout(self.dropout)(z)
        z = keras.layers.BatchNormalization()(z)
        block = keras.models.Model(inputs=block_input, outputs=z)


        block_input = keras.layers.Input(batch_shape=(input_shape[2], 1, self.nhide))
        x = keras.layers.Dense(self.nhide, activation='relu')(block_input)
        #x = keras.layers.Activation('relu')(block_input)
        x = keras.layers.Bidirectional(keras.layers.LSTM(self.binhide))(x)
        y = keras.layers.Dense(self.nhide, activation='relu')(x)
        y = keras.layers.Reshape((1, y.shape[1]))(y)
        #y = keras.layers.Activation('relu')(x)
        y = keras.layers.Bidirectional(keras.layers.LSTM(self.binhide))(y)
        z = keras.layers.add([x, y])
        z = keras.layers.Dropout(self.dropout)(z)
        z = keras.layers.BatchNormalization()(z)
        block2 = keras.models.Model(inputs=block_input, outputs=z)

        input_layer = keras.layers.Input(batch_shape=(input_shape[2], input_shape[0], input_shape[1]))
        input2 = keras.layers.Dropout(self.dropout)(input_layer)
        r = block(input2)
        r = keras.layers.Reshape((1, r.shape[1]))(r)
        r = block2(r)
        r = keras.layers.Dropout(self.dropout)(r)
        #r = keras.layers.Flatten()(r)
        #output_layer = keras.layers.Dense(output_shape, activation='relu')(r)
        output_layer = keras.layers.Dense(output_shape, activation='softmax')(r)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model
