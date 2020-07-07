import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Activation, \
    BatchNormalization, Conv1D, MaxPooling1D, Concatenate, Lambda, Reshape, UpSampling1D, GlobalAveragePooling1D, \
    Multiply, Add


class Classifier_RESNET_V2:
    def __init__(self,input_shape, nb_classes,type=1):

        if type==1:
            self.model = self.build_model(input_shape, nb_classes)
        elif type==2:
            self.model = self.build_model_without_dropout(input_shape,nb_classes)
        elif type==3:
            self.model = self.build_model_with_L2(input_shape,nb_classes)
        elif type==4:
            self.model = self.get_resnet34_se(input_shape,nb_classes)
        elif type==5:
            self.model = self.get_resnet34_BNGammaZero(input_shape,nb_classes)
        elif type==6:
            self.model = self.get_resnet34_BN2(input_shape,nb_classes)

    def build_model(self,input_shape,nb_classes):
        OUTPUT_CLASS = nb_classes
        print(input_shape)
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
            else:
                xshort = x
                # Left branch (convolutions)
            # notice the ordering of the operations has changed
            x1 = BatchNormalization(name='layer' + str(lcount))(x)
            lcount += 1
            x1 = Activation('relu')(x1)
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

        x_ecg = Flatten()(x)

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

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output')(x_ecg)

        # model = Model(inputs=input1, outputs=[out1, out2, out3, out4, out5])
        model = Model(inputs=input1, outputs=out1)

        model.summary()

        return model

    def build_model_without_dropout(self,input_shape,nb_classes):
        OUTPUT_CLASS = nb_classes  # output classes

        input1 = Input(shape=input_shape, name='input_ecg')
        k = 1  # increment every 4th residual block
        p = True  # pool toggle every other residual block (end with 2^8)
        convfilt = 64
        encoder_confilt = 64  # encoder filters' num
        convstr = 1
        ksize = 16
        poolsize = 2
        poolstr = 2

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
            else:
                xshort = x
                # Left branch (convolutions)
            # notice the ordering of the operations has changed
            x1 = BatchNormalization(name='layer' + str(lcount))(x)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Conv1D(filters=convfilt * k,
                        kernel_size=ksize,
                        padding='same',
                        strides=convstr,
                        kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = BatchNormalization(name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = Activation('relu')(x1)
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

        # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
        # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

        # Final bit
        x = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x = Activation('relu')(x)

        x_ecg = Flatten()(x)

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

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output')(x_ecg)

        model = Model(inputs=input1, outputs=out1)

        model.summary()

        return model

    def build_model_with_L2(self,input_shape,nb_classes):

        OUTPUT_CLASS = nb_classes # output classes

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

        # First convolutional block (conv,BN, relu)
        lcount = 0
        x = Conv1D(filters=convfilt,
                   kernel_size=ksize,
                   padding='same',
                   strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
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
                    strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x)
        lcount += 1
        x1 = BatchNormalization(name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
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

        x_ecg = Flatten()(x)

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

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output',
                     kernel_regularizer=keras.regularizers.l2(0.001), )(x_ecg)

        model = Model(inputs=input1, outputs=out1)

        model.summary()

        return model

    def get_resnet34_se(self,input_shape,nb_classes):

        OUTPUT_CLASS = nb_classes  # output classes

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
        se_ratio = 8

        # First convolutional block (conv,BN, relu)
        lcount = 0
        x = Conv1D(filters=convfilt,
                   kernel_size=ksize,
                   padding='same',
                   strides=convstr,
                   kernel_initializer='he_normal', name='layer' + str(lcount))(input1)
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

        x_ecg = Flatten()(x)

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output')(x_ecg)

        model = Model(inputs=input1, outputs=out1)

        return model

    def get_resnet34_BNGammaZero(self,input_shape,nb_classes):

        OUTPUT_CLASS = nb_classes  # output classes

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

        x_ecg = Flatten()(x)

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output')(x_ecg)

        model = Model(inputs=input1, outputs=out1)

        return model

    def get_resnet34_BN2(self,input_shape,nb_classes):

        OUTPUT_CLASS = nb_classes  # output classes

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

        # First convolutional block (conv,BN, relu)
        lcount = 0
        x = Conv1D(filters=convfilt,
                   kernel_size=ksize,
                   padding='same',
                   strides=convstr,
                   kernel_initializer='he_normal', name='layer' + str(lcount))(input1)
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

        x_ecg = Flatten()(x)

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output')(x_ecg)

        model = Model(inputs=input1, outputs=out1)

        return model













