import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras


# block 写法一，通过模型子类化
class block(keras.layers.Layer):
    def __init__(self, input_shape, filters, kernel_size, stride, dilation, dropout):
        super(block, self).__init__()
        self.conv1 = tfa.layers.WeightNormalization(
            keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='causal',
                dilation_rate=dilation,
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
            ))
        self.dropout1 = keras.layers.Dropout(dropout)
        self.conv2 = tfa.layers.WeightNormalization(
            keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='causal',
                dilation_rate=dilation,
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
            ))
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


class TCN:
    def __init__(self, input_shape, output_shape):
        self.stride = 1
        self.dropout = 0.2
        self.filters = [32, 32]
        self.num_levels = 2
        self.kernel_size = 3

        self.model = self.build_model(input_shape, output_shape)


    # block写法二，通过内部函数
    def build_block(self, input_shape, filters, kernel_size, stride, dilation, dropout):
        input_layer = keras.layers.Input(shape=(input_shape[0], input_shape[1]))

        # 注意：在TCN源码中使用了WeightNormalization层，
        # 在tensorflow中引入该层时可以正常训练，但是在保存模型时会有奇怪的错误
        # 该写法符合官方给的代码示例，因此可能是与callback不兼容引起的(未解决)
        # 参考资料：https://www.tensorflow.org/addons/tutorials/layers_weightnormalization

        '''
        conv1 = tfa.layers.WeightNormalization(
            keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='causal',
                dilation_rate=dilation,
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
            ))(input_layer)

        '''
        conv1 = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='causal',
                dilation_rate=dilation,
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
            )(input_layer)

        dropout1 = keras.layers.Dropout(dropout)(conv1)

        '''
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
        '''
        conv2 = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding='causal',
                dilation_rate=dilation,
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(0, 0.01)
            )(dropout1)

        out = keras.layers.Dropout(dropout)(conv2)
        res = keras.layers.Conv1D(filters=filters, kernel_size=1)(input_layer)
        out2 = keras.layers.add([out, res])
        output_layer = keras.layers.Activation('relu')(out2)
        block = keras.models.Model(inputs=input_layer, outputs=output_layer)
        return block



    def build_model(self, input_shape, output_shape):
        # 写法一

        model = keras.models.Sequential()
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_shape = [input_shape[0], 0]
            in_shape[1] = input_shape[1] if i == 0 else self.filters[i-1]
            filters = self.filters[i]
            block = self.build_block(in_shape, filters, self.kernel_size, self.stride, dilation_size, self.dropout)
            model.add(block)
        model.add(keras.layers.Flatten())
        #model.add(keras.layers.Dense(output_shape, activation='sigmoid'))
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
        return model



        # 写法二
        '''
        block1 = self.build_block(input_shape, 32, self.kernel_size, self.stride, 1, self.dropout)
        block2 = self.build_block([input_shape[0], 32], 32, self.kernel_size, self.stride, 2, self.dropout)

        #block1 = block(input_shape, 32, self.kernel_size, self.stride, 1, self.dropout)
        #block2 = block([input_shape[0], 32], 32, self.kernel_size, self.stride, 2, self.dropout)



        input_layer = keras.layers.Input(shape=(input_shape[0], input_shape[1]))
        out = block1(input_layer)
        out = block2(out)
        out = keras.layers.Flatten()(out)
        output_layer = keras.layers.Dense(output_shape, activation='sigmoid')(out)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model
        '''

