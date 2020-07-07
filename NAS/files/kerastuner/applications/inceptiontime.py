import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from kerastuner.engine import hypermodel
from .metric import f1, recall, precision

class Hyperinceptiontime(hypermodel.HyperModel):
    """An inceptiontime HyperModel.

        # Arguments:

            include_top: whether to include the fully-connected
                layer at the top of the network.
            input_shape: Optional shape tuple, e.g. `(256, 3)`.
                  One of `input_shape` or `input_tensor` must be
                  specified.
            input_tensor: Optional Keras tensor (i.e. output of
                `layers.Input()`) to use as image input for the model.
                  One of `input_shape` or `input_tensor` must be
                  specified.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True,
                and if no `weights` argument is specified.
            **kwargs: Additional keyword arguments that apply to all
                HyperModels. See `kerastuner.HyperModel`.
        """
    def __init__(self,
                 include_top=True,
                 input_shape=None,
                 input_tensor=None,
                 classes=None,
                 **kwargs):
        super(Hyperinceptiontime, self).__init__(**kwargs)
        if include_top and classes is None:
            raise ValueError('You must specify `classes` when '
                             '`include_top=True`')

        if input_shape is None and input_tensor is None:
            raise ValueError('You must specify either `input_shape` '
                             'or `input_tensor`.')

        self.include_top = include_top
        self.input_shape = input_shape
        self.input_tensor = input_tensor
        self.classes = classes

    def build(self, hp):
        # Model definition.
        if self.input_tensor is not None:
            inputs = tf.keras.utils.get_source_inputs(self.input_tensor)
            input_layer = self.input_tensor
        else:
            inputs = layers.Input(shape=self.input_shape)
            input_layer = inputs

        type = hp.Choice('type', ['v1', 'v2'], default='v1')
        depth = hp.Choice('depth', [6, 9, 12], default=6)
        nb_filters = hp.Choice('nb_filters', [16, 32, 64], default=32)
        kernel_size = hp.Choice('kernel_size', [20, 40, 60, 80], default=40)
        use_residual = True
        bottleneck_size = hp.Choice('bottleneck_size', [16, 32, 64], default=32)

        x = input_layer
        input_res = input_layer

        if type == 'v1':
            for d in range(depth):

                x = inception_module(x,
                                     nb_filters=nb_filters,
                                     kernel_size=kernel_size,
                                     bottleneck_size=bottleneck_size
                                     )

                if use_residual and d % 3 == 2:
                    x = shortcut_layer(input_res, x)
                    input_res = x
        else:
            for d in range(depth):

                x = inception_module(x,
                                     nb_filters=nb_filters,
                                     kernel_size=kernel_size,
                                     bottleneck_size=bottleneck_size,
                                     stride=2)

                x = inception_module(x,
                                     nb_filters=nb_filters,
                                     kernel_size=kernel_size,
                                     bottleneck_size=bottleneck_size,
                                     stride=1)

                if use_residual and d % 3 == 2:
                    x = shortcut_layer(input_res, x, stride=8)
                    input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        if self.include_top:
            output_layer = layers.Dense(
                self.classes, activation='softmax')(gap_layer)
            model = keras.Model(inputs=input_layer, outputs=output_layer, name='Inceptiontime')
            optimizer_name = hp.Choice(
                'optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
            optimizer = keras.optimizers.get(optimizer_name)
            optimizer.learning_rate = hp.Choice(
                'learning_rate', [0.1, 0.01, 0.001], default=0.01)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy',precision, recall, f1])
            return model
        else:
            return keras.Model(inputs=input_layer, outputs=gap_layer, name='Inceptiontime')

def inception_module(input_tensor, nb_filters=32, kernel_size=40, use_bottleneck=True,bottleneck_size=32, stride=1, activation='linear'):

        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        # [40,20,10]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = layers.Conv1D(filters=nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = layers.Concatenate(axis=2)(conv_list)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        return x

def shortcut_layer(input_tensor, out_tensor, stride=1 ):
        shortcut_y = layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same',strides=stride, use_bias=False)(input_tensor)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x