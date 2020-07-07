import tensorflow.keras as keras
import  os
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils.uts_classification.utils import readmts_uci_har,transform_labels
from utils.AFClassication.data import loaddata
from tools import AdvancedLearnignRateScheduler
from models.classification.inception import Classifier_INCEPTION

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载UCI_HAR_Dataset
(X_train, y_train), (Xval, yval), (final_testset, final_testtarget) , (R_train, Rval, Rtest), (
    P_train, Pval, Ptest), (Q_train, Qval, Qtest), (T_train, Tval, Ttest)= loaddata()

# x_train = np.concatenate((X_train[0], Xval[0]), axis=0)
# y_train = np.concatenate((y_train, yval), axis=0)
shape = X_train[0].shape
if len(shape) == 2:
    X_train[0] = np.expand_dims(X_train[0], axis=2)
    Xval[0] = np.expand_dims(Xval[0], axis=2)
    final_testset[0] = np.expand_dims(final_testset[0], axis=2)

NUM_CLASSES = 3
# y_train = to_categorical(y_train, NUM_CLASSES)
# y_test = to_categorical(y_test, NUM_CLASSES)

# train_number = int(x_train.shape[0]/ 16) * 16
# x_train = x_train[0:train_number]
# y_train = y_train[0:train_number]

train_number = int(len(X_train[0]) / 16) * 16
val_number = int(len(Xval[0]) / 16) * 16

print(X_train[0].shape)
print(y_train.shape)
print(final_testset[0].shape)
print(final_testtarget.shape)
print(y_train[:3])

input_shape = X_train[0].shape[1:]
nb_classes = 3

model = Classifier_INCEPTION(input_shape,nb_classes,'inceptiontime_v2').model
adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',
                  metrics=['accuracy'])
model.summary()

model.fit(X_train[0][0:train_number], y_train[0:train_number], epochs=100, validation_data=(Xval[0][0:val_number],yval[0:val_number]), batch_size=16, callbacks=[keras.callbacks.EarlyStopping(patience=10)])