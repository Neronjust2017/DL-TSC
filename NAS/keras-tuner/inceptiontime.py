import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from kerastuner.applications.inceptiontime import Hyperinceptiontime
from kerastuner import RandomSearch, Hyperband
from utils.uts_classification.utils import readmts_uci_har,transform_labels
import numpy as  np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载UCI_HAR_Dataset
file_name = '../../datasets/mts_data/UCI_HAR_Dataset'
x_train, y_train, x_test, y_test = readmts_uci_har(file_name)
data = np.concatenate((x_train, x_test),axis=0)
label = np.concatenate((y_train, y_test),axis=0)
N = data.shape[0]
ind = int(N*0.9)
x_train = data[:ind]
y_train = label[:ind]
x_test = data[ind:]
y_test = label[ind:]
y_train, y_test = transform_labels(y_train, y_test)
NUM_CLASSES = 6
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train[:3])

# Import an hypertunable version of Inceptiontime.
hypermodel = Hyperinceptiontime(
    input_shape=x_train.shape[1:],
    classes=NUM_CLASSES)

# Initialize the hypertuner

# tuner = RandomSearch(
#     hypermodel,
#     objective='val_loss',
#     max_trials=2,
#     project_name='AF_inceptiontime',
#     directory='test_directory')

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=20,
    # hyperband_iterations=20,
    project_name='UCI_inceptiontime',
    directory='nas_result')

# Display search overview.
tuner.search_space_summary()

# Performs the hypertuning.
tuner.search(x_train, y_train, epochs=100, validation_split=0.1,batch_size=128,
             callbacks=[keras.callbacks.EarlyStopping(patience=10)])

# Show the best models, their hyperparameters, and the resulting metrics.
tuner.results_summary()

# Retrieve the best model.
best_models = tuner.get_best_models(num_models=10)

# Evaluate the best model.
for i in range(10):
    loss, accuracy, precision, recall, f1  = best_models[i].evaluate(x_test, y_test)
    print('*************************----best_model_' + str(i) + '----*************************')
    print('loss:', loss)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

