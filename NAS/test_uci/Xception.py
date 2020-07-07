import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)
rootPath = curPath
for i in range(2):
    rootPath = os.path.split(rootPath)[0]
print(rootPath)
sys.path.append(rootPath)

import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from utils.uts_classification.utils import readmts_uci_har,transform_labels
import autokeras as ak
import numpy as  np
from NAS.logger import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

classes = ['Cla.1','Cla.2','Cla.3','Cla.4','Cla.5','Cla.6']
NUM_CLASSES = 6
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train[:3])

# 指定参数
name = 'Xception_Greedy'
max_trials = 30
if not os.path.isdir('./result/'+name):
    os.makedirs('./result/'+name)

# Initialize the classifier.
input_node = ak.Input()
output_node = ak.Xception1dBlock()(input_node)
output_node = ak.ClassificationHead(num_classes=6,dropout_rate=0)(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=max_trials,
                   name=name,directory='result')

clf.tuner.executions_per_trial = 2
# Logger for NAS
fp = open('result/'+name+'/log','w')
fp.close()
fp = open('result/'+name+'/log_file','w')
fp.close()
sys.stdout = Logger('result/'+name+'/log', sys.stdout)
sys.stderr = Logger('result/'+name+'/log_file', sys.stderr)		# redirect std err, if necessary

clf.tuner.search_space_summary()
# Search for the best model.
clf.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=128, callbacks=[keras.callbacks.EarlyStopping(patience=10)],verbose=1)

clf.tuner.results_summary()

# Evaluate the best model
best_model = clf.tuner.get_best_model()[1]
print('*************************----best_model----*************************')
cvconfusion = np.zeros((NUM_CLASSES, NUM_CLASSES))
ypred = best_model.predict(x_test)
ypred = np.argmax(ypred, axis=1)
ytrue = np.argmax(y_test, axis=1)
cvconfusion[:, :] = confusion_matrix(ytrue, ypred)
F1 = np.zeros((6, 1))
Precision = np.zeros((6, 1))
Recall = np.zeros((6, 1))
Accuracy = 0
for i in range(6):
    F1[i] = 2 * cvconfusion[i, i] / (
            np.sum(cvconfusion[i, :]) + np.sum(cvconfusion[:, i]))
    print("test F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
    Precision[i] = cvconfusion[i, i] / np.sum(cvconfusion[:, i])
    Recall[i] = cvconfusion[i, i] / np.sum(cvconfusion[i, :])
    Accuracy += cvconfusion[i, i] / np.sum(cvconfusion[:, :])
print("test Overall F1 measure: {:1.4f}".format(np.mean(F1[0:6])))

# Evaluate the best 10 models( only a convenience shortcut, recommended to retrain the models)
best_models = clf.tuner.get_best_models(num_models=15)
for j in range(15):
    print('*************************----best_model_'+str(j)+'----*************************')
    model = best_models[j][2]
    cvconfusion = np.zeros((NUM_CLASSES, NUM_CLASSES))
    ypred = model.predict(x_test)
    ypred = np.argmax(ypred, axis=1)
    ytrue = np.argmax(y_test, axis=1)
    cvconfusion[:, :] = confusion_matrix(ytrue, ypred)
    F1 = np.zeros((6, 1))
    Precision = np.zeros((6, 1))
    Recall = np.zeros((6, 1))
    Accuracy = 0
    for i in range(6):
        F1[i] = 2 * cvconfusion[i, i] / (
                np.sum(cvconfusion[i, :]) + np.sum(cvconfusion[:, i]))
        print("test F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
        Precision[i] = cvconfusion[i, i] / np.sum(cvconfusion[:, i])
        Recall[i] = cvconfusion[i, i] / np.sum(cvconfusion[i, :])
        Accuracy += cvconfusion[i, i] / np.sum(cvconfusion[:, :])
    print("test Overall F1 measure: {:1.4f}".format(np.mean(F1[0:6])))
