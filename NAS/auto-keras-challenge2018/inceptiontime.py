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
from utils.AFClassication.data_challenge2018 import loaddata

import autokeras as ak
import numpy as  np
from NAS.logger import Logger
from tools import AdvancedLearnignRateScheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
(X_train, y_train), (Xval, yval), (final_testset, final_testtarget)= loaddata()
print(X_train.shape)
print(y_train.shape)
print(final_testset.shape)
print(final_testtarget.shape)
print(y_train[:3])

# Initialize the classifier.
input_node = ak.Input()
output_node = ak.InceptionTimeBlock(type='v2')(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=20,
                   name='2018_inceptiontime_Greedy',directory='nas_result')
#
sys.stdout = Logger('nas_result/2018_inceptiontime_Greedy/log', sys.stdout)
sys.stderr = Logger('nas_result/2018_inceptiontime_Greedy/log_file', sys.stderr)		# redirect std err, if necessary

clf.tuner.search_space_summary()
# Search for the best model.

clf.fit(X_train,y_train, epochs=100, validation_data=(Xval, yval), batch_size=16, callbacks=[keras.callbacks.EarlyStopping(patience=10),
    AdvancedLearnignRateScheduler(monitor='val_loss', patience=6, verbose=1, mode='auto', decayRatio=0.1, warmup_batches=5, init_lr=0.001)],verbose=0)

clf.tuner.results_summary()

# Evaluate the best model on the testing data.
# loss, accuracy, precision, recall, f1 = clf.evaluate(final_testset, final_testtarget)
# print('*************************----best_model----*************************')
# print('loss:', loss)
# print('accuracy:', accuracy)
# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)

# Evaluate the best 10 models( only a convenience shortcut, recommended to retrain the models)
best_models = clf.tuner.get_best_models(num_models=10)

for i in range(10):
    loss, accuracy, precision, recall, f1 = best_models[i][2].evaluate(final_testset, final_testtarget)

    print('*************************----best_model_'+str(i)+'----*************************')
    print('loss:', loss)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)


# model = clf.export_model()
# model.save('nas_result/2018_inceptiontime_Greedy/bestmodel.h5')