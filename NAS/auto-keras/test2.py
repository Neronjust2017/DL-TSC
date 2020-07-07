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

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载UCI_HAR_Dataset
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(y_train[:3]) # array([7, 2, 1], dtype=uint8)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train[:3])

x_train = x_train[:600,:,:]
x_test  = x_test[:100,:,:]
y_train = y_train[:600]
y_test = y_test[:100]
# Initialize the classifier.
input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation()(output_node)
output_node1 = ak.ConvBlock()(output_node)
output_node = ak.ClassificationHead()(output_node)

clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=2,
                   name='test',directory='nas_result')

sys.stdout = Logger('nas_result/test/log', sys.stdout)
sys.stderr = Logger('nas_result/test/log_file', sys.stderr)		# redirect std err, if necessary

clf.tuner.search_space_summary()
# Search for the best model.
clf.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=128, callbacks=[keras.callbacks.EarlyStopping(patience=10)])



clf.tuner.results_summary()

# Evaluate the best model on the testing data.
loss, accuracy = clf.evaluate(x_test, y_test)
print('*************************----best_model----*************************')
print('loss:', loss)
print('accuracy:', accuracy)


# Evaluate the best 10 models( only a convenience shortcut, recommended to retrain the models)
best_models = clf.tuner.get_best_models(num_models=10)

for i in range(2):
    loss, accuracy= best_models[i][2].evaluate(x_test, y_test)

    print('*************************----best_model_'+str(i)+'----*************************')
    print('loss:', loss)
    print('accuracy:', accuracy)


model = clf.export_model()
model.save('nas_result/test/bestmodel.h5')
