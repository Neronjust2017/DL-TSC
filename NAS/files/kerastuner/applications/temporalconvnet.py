import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from kerastuner.engine import hypermodel
from .metric import f1, recall, precision

class HyperTemporalConvNet(hypermodel.HyperModel):
    """An TemporalConvNet HyperModel.

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
        super(HyperTemporalConvNet, self).__init__(**kwargs)
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

        stride = 1
        dropout = hp.Float('drop', 0, 0.6, step=0.1, default=0.0)
        kernel_size = hp.Choice('kernel_size', [16, 32, 64], default=32)

        num_levels = hp.Choice('num_levels', [2, 4, 8], default=2)
        filters = hp.Choice('filters',[16, 32 , 64],default=32)
        for i in range(num_levels):
            dilation_size = 2 ** i
            if i==0:
                block = TemporalBlock_1(input_layer, filters, kernel_size, stride, dilation_size, dropout)
            else:
                block = TemporalBlock_1(block, filters, kernel_size, stride, dilation_size, dropout)
        out = layers.Lambda(lambda x: x[:,-1,:])(block)

        if self.include_top:
            output_layer = layers.Dense(
                self.classes, activation='softmax')(out)
            model = keras.Model(inputs=input_layer, outputs=output_layer, name='TemporalConvNet')
            optimizer_name = hp.Choice(
                'optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
            optimizer = keras.optimizers.get(optimizer_name)
            optimizer.learning_rate = hp.Choice(
                'learning_rate', [0.01, 0.001], default=0.01)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', precision, recall, f1])
            return model
        else:
            return keras.Model(inputs=input_layer, outputs=out, name='TemporalConvNet')

def TemporalBlock_1(input_tensor,filters, kernel_size, stride, dilation, dropout):

    conv1 =tfa.layers.WeightNormalization(
        keras.layers.Conv1D(
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
    return out2
