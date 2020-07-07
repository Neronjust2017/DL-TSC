import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Activation, \
    BatchNormalization, Conv1D, MaxPooling1D, Concatenate, Lambda, Reshape, UpSampling1D, concatenate, add
# from tensorflow.keras.layers.merge import concatenate, add
import tensorflow.keras.backend as K


class Classifier_RESNEXT:
    def __init__(self,input_shape, nb_classes):

        self.model = self.build_model(input_shape, nb_classes)

    def build_model(self,input_shape,nb_classes):
        # refercen: https://github.com/titu1994/Keras-ResNeXt/blob/master/resnext.py
        OUTPUT_CLASS = nb_classes
        input1 = Input(shape=input_shape, name='input_ecg')
        k = 1  # increment every 4th residual block
        p = True  # pool toggle every other residual block (end with 2^8)
        convfilt = 64
        encoder_confilt = 64  # encoder filters' num
        convstr = 1
        ksize = 16
        poolsize = 2
        poolstr = 2
        drop = 0.5
        cardinality = 16
        grouped_channels = int(convfilt / cardinality)

        # First convolutional block (conv,BN, relu)
        lcount = 0
        x = Conv1D(filters=convfilt,
                   kernel_size=ksize,
                   padding='same',
                   strides=convstr,
                   kernel_initializer='he_normal', name='layer' + str(lcount))(input1)
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
        for l in range(15):

            if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
                k += 1
                # increase depth by 1x1 Convolution case dimension shall change
                xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
                lcount += 1
                x = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
                lcount += 1
            else:
                xshort = x
                # Left branch (convolutions)

            grouped_channels = int(convfilt * k / cardinality)
            # notice the ordering of the operations has changed
            x1 = BatchNormalization(name='layer' + str(lcount))(x)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Dropout(drop)(x1)
            ### grouped convlutional block
            ksize_choice = [2, 4, 8, 16, 20, 24, 28, 32]
            group_list = []
            for c in range(cardinality):
                x_tmp = Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels])(x1)

                x_tmp = Conv1D(grouped_channels, ksize_choice[int(c / 2)], padding='same', strides=convstr,
                               kernel_initializer='he_normal', name='layer' + str(lcount))(x_tmp)
                group_list.append(x_tmp)
                lcount += 1

            x1 = concatenate(group_list, axis=-1)
            # x1 = x_tmp

            # x1 = Conv1D(filters=convfilt * k,
            #             kernel_size=ksize,
            #             padding='same',
            #             strides=convstr,
            #             kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            # lcount += 1
            x1 = BatchNormalization(name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Dropout(drop)(x1)

            ### grouped convlutional block
            group_list = []
            for c in range(cardinality):
                x_tmp = Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels])(x1)

                x_tmp = Conv1D(grouped_channels, ksize_choice[int(c / 2)], padding='same', strides=convstr,
                               kernel_initializer='he_normal', name='layer' + str(lcount))(x_tmp)

                group_list.append(x_tmp)
                lcount += 1

            x1 = concatenate(group_list, axis=-1)
            # x1 = x_tmp
            # x1 = Conv1D(filters=convfilt * k,
            #             kernel_size=ksize,
            #             padding='same',
            #             strides=convstr,
            #             kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            # lcount += 1
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

        x_ecg = Flatten()(x)

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output')(x_ecg)

        model = Model(inputs=input1, outputs=out1)

        model.summary()

        return model


if __name__ =='__main__':
    model = Classifier_RESNEXT((18176,1),3).model










