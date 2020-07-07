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
from tensorflow.keras.utils import to_categorical
from utils.uts_classification.utils import readmts_uci_har,transform_labels
import autokeras as ak
import numpy as  np
from NAS.logger import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

input_node = ak.Input()
output_node = ak.ResNetV2Block()(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=30,
                   name='UCI_resnet_v2_Greedy',directory='nas_result')

sys.stdout = Logger('nas_result/UCI_resnet_v2_Greedy/log', sys.stdout)
sys.stderr = Logger('nas_result/UCI_resnet_v2_Greedy/log_file', sys.stderr)		# redirect std err, if necessary

clf.tuner.search_space_summary()
# Search for the best model.
clf.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=128, callbacks=[keras.callbacks.EarlyStopping(patience=10)])

clf.tuner.results_summary()

# Evaluate best model on the testing data.
loss, accuracy, precision, recall, f1 = clf.evaluate(x_test, y_test)
print('*************************----best_model----*************************')
print('loss:', loss)
print('accuracy:', accuracy)
print('precision:', precision)
print('recall:', recall)
print('f1:', f1)

# Evaluate best 10 models( only a convenience shortcut, recommended to retrain the models)
best_models = clf.tuner.get_best_models(num_models=10)

for i in range(10):
    loss, accuracy, precision, recall, f1 = best_models[i][2].evaluate(x_test, y_test)

    print('*************************----best_model_'+str(i)+'----*************************')
    print('loss:', loss)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)


model = clf.export_model()
model.save('nas_result/UCI_resnet_v2_Greedy/bestmodel.h5')