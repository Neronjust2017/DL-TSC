import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Activation, \
    BatchNormalization, Conv1D, MaxPooling1D, Concatenate, Lambda, Reshape, UpSampling1D, GlobalAveragePooling1D, \
    Multiply, Add

from kerastuner.engine import hypermodel
from .metric import f1, recall, precision


class HyperResNetV2(hypermodel.HyperModel):
    """A ResNet_V2 HyperModel.

    # Arguments:

        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
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
        super(HyperResNetV2, self).__init__(**kwargs)
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
            x = self.input_tensor
        else:
            inputs = layers.Input(shape=self.input_shape)
            x = inputs

        type = hp.Choice('type', ['resnet34','resnet34_with_L2','resnet34_se', 'resnet34_BNGammaZero', 'resnet34_BN2'],
                         default='resnet34')
        convfilt = hp.Choice('convfilt', [16, 32, 64], default=32)
        ksize = hp.Choice('ksize', [8, 16, 32, 64], default=16)
        depth = hp.Choice('depth', [7, 15, 23, 31], default=15)
        drop = hp.Float('drop', 0.0, 0.6, step=0.1, default=0.0)


        if type=='resnet34':

            x = build_model(x, convfilt, ksize, depth, drop)

        elif type=='resnet34_with_L2':

            x = build_model_with_L2(x, convfilt, ksize, depth, drop)

        elif type=='resnet34_se':

            x = get_resnet34_se(x, convfilt, ksize, depth, drop)

        elif type=='resnet34_BNGammaZero':
            x = get_resnet34_BNGammaZero(x, convfilt, ksize, depth, drop)

        elif type=='resnet34_BN2':
            x = get_resnet34_BN2(x, convfilt, ksize, depth, drop)

        if self.include_top:
            if type=='resnet34_with_L2':
                x = layers.Dense(
                     self.classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001) )(x)
            else:
                x = layers.Dense(
                    self.classes, activation='softmax')(x)

            model = keras.Model(inputs=inputs, outputs=x, name='Resnet_v2')
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
            return keras.Model(inputs=inputs, outputs=x, name='Resnet_v2')
            

def build_model(input_tensor, convfilt=64, ksize=16, depth=15, drop=0):

    k = 1  # increment every 4th residual block
    p = True  # pool toggle every other residual block (end with 2^8)
    convfilt = convfilt
    encoder_confilt = 64  # encoder filters' num
    convstr = 1
    ksize = ksize
    poolsize = 2
    poolstr = 2
    drop = 0.5
    depth = depth

    # First convolutional block (conv,BN, relu)
    lcount = 0
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal', name='layer' + str(lcount))(input_tensor)
    lcount += 1
    x = BatchNormalization(name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr,
                kernel_initializer='he_normal', name='layer' + str(lcount))(x)
    lcount += 1
    x1 = BatchNormalization(name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = Activation('relu')(x1)
    if drop:
        x1 = Dropout(drop)(x1)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr,
                kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1, x2

    fms = []
    ## Main loop
    p = not p
    for l in range(depth):

        if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
            k += 1
            # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
            lcount += 1
        else:
            xshort = x
            # Left branch (convolutions)
        # notice the ordering of the operations has changed
        x1 = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = BatchNormalization(name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        if p:
            x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)

            # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
        else:
            x2 = xshort  # pool or identity
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p  # toggle pooling
        # if l == 5:
        #     fms.append(x)
        # if l == 6:
        #     fms.append(x)
        #     fms.append(x)
        #     fms.append(x)
    # fms的内容：[<tf.Tensor 'add_6/Identity:0' shape=(None, 1136, 128) dtype=float32>,
    #             <tf.Tensor 'add_7/Identity:0' shape=(None, 1136, 128) dtype=float32>,
    #             <tf.Tensor 'add_7/Identity:0' shape=(None, 1136, 128) dtype=float32>,
    #             <tf.Tensor 'add_7/Identity:0' shape=(None, 1136, 128) dtype=float32>]

    # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
    # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

    # Final bit
    x = BatchNormalization(name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    x = Flatten()(x)

    # bbox_num = 1
    #
    # x2od2 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
    #                kernel_initializer='he_normal')(
    #     fms[0])
    # out2 = Reshape((1136, bbox_num, 2), name='aux_output1')(x2od2)
    #
    # x2od3 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
    #                kernel_initializer='he_normal')(
    #     fms[1])
    # out3 = Reshape((1136, bbox_num, 2), name='aux_output2')(x2od3)
    #
    # x2od4 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
    #                kernel_initializer='he_normal')(
    #     fms[2])
    # out4 = Reshape((1136, bbox_num, 2), name='aux_output3')(x2od4)
    #
    # x2od5 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
    #                kernel_initializer='he_normal')(
    #     fms[3])
    # out5 = Reshape((1136, bbox_num, 2), name='aux_output4')(x2od5)

    return x

def build_model_with_L2(input_tensor, convfilt=64, ksize=16, depth=15, drop=0):

    k = 1  # increment every 4th residual block
    p = True  # pool toggle every other residual block (end with 2^8)
    convfilt = convfilt
    encoder_confilt = 64  # encoder filters' num
    convstr = 1
    ksize = ksize
    poolsize = 2
    poolstr = 2
    drop = drop
    depth = depth

    # First convolutional block (conv,BN, relu)
    lcount = 0
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
               kernel_initializer='he_normal', name='layer' + str(lcount))(input_tensor)
    lcount += 1
    x = BatchNormalization(name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                kernel_initializer='he_normal', name='layer' + str(lcount))(x)
    lcount += 1
    x1 = BatchNormalization(name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = Activation('relu')(x1)
    if drop:
        x1 = Dropout(drop)(x1)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1, x2

    # fms = []
    ## Main loop
    p = not p
    for l in range(depth):

        if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
            k += 1
            # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
            lcount += 1
        else:
            xshort = x
            # Left branch (convolutions)
        # notice the ordering of the operations has changed
        x1 = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = BatchNormalization(name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        if p:
            x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)

            # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
        else:
            x2 = xshort  # pool or identity
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p  # toggle pooling
        # if l == 5:
        #     fms.append(x)
        # if l == 6:
        #     fms.append(x)
        #     fms.append(x)
        #     fms.append(x)

    # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
    # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

    # Final bit
    x = BatchNormalization(name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    x = Flatten()(x)

    # bbox_num = 1
    #
    # x2od2 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
    #                kernel_initializer='he_normal')(
    #     fms[0])
    # out2 = Reshape((1136, bbox_num, 2), name='aux_output1')(x2od2)
    #
    # x2od3 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
    #                kernel_initializer='he_normal')(
    #     fms[1])
    # out3 = Reshape((1136, bbox_num, 2), name='aux_output2')(x2od3)
    #
    # x2od4 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
    #                kernel_initializer='he_normal')(
    #     fms[2])
    # out4 = Reshape((1136, bbox_num, 2), name='aux_output3')(x2od4)
    #
    # x2od5 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
    #                kernel_initializer='he_normal')(
    #     fms[3])
    # out5 = Reshape((1136, bbox_num, 2), name='aux_output4')(x2od5)

    return x

def get_resnet34_se(input_tensor, convfilt=64, ksize=16, depth=15, drop=0):

    k = 1  # increment every 4th residual block
    p = True  # pool toggle every other residual block (end with 2^8)
    convfilt = convfilt
    encoder_confilt = 64  # encoder filters' num
    convstr = 1
    ksize = ksize
    poolsize = 2
    poolstr = 2
    depth = depth
    drop = drop
    se_ratio = 8

    # First convolutional block (conv,BN, relu)
    lcount = 0
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal', name='layer' + str(lcount))(input_tensor)
    lcount += 1
    squeeze = GlobalAveragePooling1D()(x)
    excitation = Dense(convfilt * k // se_ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(convfilt * k)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, convfilt * k))(excitation)
    x = Multiply()([x, excitation])
    x = BatchNormalization(name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr,
                kernel_initializer='he_normal', name='layer' + str(lcount))(x)
    lcount += 1
    x1 = BatchNormalization(name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = Activation('relu')(x1)
    if drop:
        x1 = Dropout(drop)(x1)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr,
                kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
    lcount += 1

    squeeze = GlobalAveragePooling1D()(x1)
    excitation = Dense(convfilt * k // se_ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(convfilt * k)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, convfilt * k))(excitation)
    x1 = Multiply()([x1, excitation])

    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1, x2

    ## Main loop
    p = not p
    for l in range(depth):

        if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
            k += 1
            # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
            lcount += 1
        else:
            xshort = x
            # Left branch (convolutions)
        # notice the ordering of the operations has changed

        x1 = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = BatchNormalization(name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1

        squeeze = GlobalAveragePooling1D()(x1)
        excitation = Dense(convfilt * k // se_ratio)(squeeze)
        excitation = Activation('relu')(excitation)
        excitation = Dense(convfilt * k)(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape((1, convfilt * k))(excitation)
        x1 = Multiply()([x1, excitation])

        if p:
            x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)

            # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
        else:
            x2 = xshort  # pool or identity
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p  # toggle pooling

    # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
    # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

    # Final bit
    x = BatchNormalization(name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    x = Flatten()(x)

    return x

def get_resnet34_BNGammaZero(input_tensor, convfilt=64, ksize=16, depth=15, drop=0):

    k = 1  # increment every 4th residual block
    p = True  # pool toggle every other residual block (end with 2^8)
    convfilt = convfilt
    encoder_confilt = 64  # encoder filters' num
    convstr = 1
    ksize = ksize
    poolsize = 2
    poolstr = 2
    depth = depth
    drop = drop

    # First convolutional block (conv,BN, relu)
    lcount = 0
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal', name='layer' + str(lcount))(input_tensor)
    lcount += 1
    x = BatchNormalization(name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr,
                kernel_initializer='he_normal', name='layer' + str(lcount))(x)
    lcount += 1
    x1 = BatchNormalization(name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = Activation('relu')(x1)
    if drop:
        x1 = Dropout(drop)(x1)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr,
                kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1, x2

    ## Main loop
    p = not p
    for l in range(15):

        if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
            k += 1
            # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
            lcount += 1
        else:
            xshort = x
            # Left branch (convolutions)
        # notice the ordering of the operations has changed
        x1 = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = BatchNormalization(name='layer' + str(lcount), gamma_initializer='zeros')(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        if p:
            x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)

            # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
        else:
            x2 = xshort  # pool or identity
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p  # toggle pooling

    # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
    # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

    # Final bit
    x = BatchNormalization(name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    x = Flatten()(x)

    return x

def get_resnet34_BN2(input_tensor, convfilt=64, ksize=16, depth=15, drop=0):

    k = 1  # increment every 4th residual block
    p = True  # pool toggle every other residual block (end with 2^8)
    convfilt = convfilt
    encoder_confilt = 64  # encoder filters' num
    convstr = 1
    ksize = ksize
    poolsize = 2
    poolstr = 2
    depth = depth
    drop = drop

    # First convolutional block (conv,BN, relu)
    lcount = 0
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal', name='layer' + str(lcount))(input_tensor)
    lcount += 1
    x = BatchNormalization(axis=1, name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr,
                kernel_initializer='he_normal', name='layer' + str(lcount))(x)
    lcount += 1
    x1 = BatchNormalization(axis=1, name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = Activation('relu')(x1)
    if drop:
        x1 = Dropout(drop)(x1)
    x1 = Conv1D(filters=convfilt,
                kernel_size=ksize,
                padding='same',
                strides=convstr,
                kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
    lcount += 1
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr, padding='same')(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1, x2

    ## Main loop
    p = not p
    for l in range(15):

        if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
            k += 1
            # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
            lcount += 1
        else:
            xshort = x
            # Left branch (convolutions)
        # notice the ordering of the operations has changed
        x1 = BatchNormalization(axis=1, name='layer' + str(lcount))(x)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = BatchNormalization(axis=1, name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
        if drop:
            x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt * k,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        if p:
            x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)

            # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
        else:
            x2 = xshort  # pool or identity
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p  # toggle pooling

    # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
    # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

    # Final bit
    x = BatchNormalization(axis=1, name='layer' + str(lcount))(x)
    lcount += 1
    x = Activation('relu')(x)

    x = Flatten()(x)

    return x