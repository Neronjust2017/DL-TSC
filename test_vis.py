from comet_ml import experiment
from data_loader.uts_classification_data_loader import UtsClassificationDataLoader
from models.uts_classification_model import UtsClassificationModel
from trainers.uts_classification_trainer import UtsClassificationTrainer
from evaluater.uts_classification_evaluater import UtsClassificationEvaluater
from utils.config import process_config_VisOverfit
from utils.dirs import create_dirs
from utils.utils import get_args
import pandas as pd
import numpy as  np
import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns

def plot_metrics_matrix():
    metrics = np.arange(9).reshape(3,3)
    col_labels = []
    for i in range(3):
        col_labels.append("Cls."+str(i))
    row_labels = ['Precision', 'Recall', 'F1-score']
    sns.set(font_scale=2.5)
    f, hm = plt.subplots(figsize=(25, 25))
    hm = sns.heatmap(metrics,
                     xticklabels=True,
                     yticklabels=True,
                     cmap="YlGnBu",
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 20})
    hm.set_xticklabels(col_labels, fontsize=18, horizontalalignment='right')
    hm.set_yticklabels(row_labels, fontsize=18, horizontalalignment='right')
    plt.title('Metrics')
    plt.show()
    plt.savefig('hhh.png', format='png')
    plt.cla()
    plt.clf()
    plt.close('all')

def plot_trainingsize_metric2(data):

    plt.figure()
    plt.plot(data["training_size"],data["accuracy"])
    plt.plot(data["training_size"], data["f1"])
    plt.title('model ' )
    plt.ylabel('metric', fontsize='large')
    plt.xlabel('training_size', fontsize='large')
    plt.legend(['accuracy', 'f1'], loc='upper left')
    plt.show()
    plt.savefig('hhhh.png', bbox_inches='tight')
def plot_trainingsize_metric1(data):

    plt.figure(num=3)
    plt.show()
    plt.plot(data["training_size"],data["accuracy"])
    plt.plot(data["training_size"], data["f1"])
    plt.show()
    plt.title('model ' )
    plt.ylabel('metric', fontsize='large')
    plt.xlabel('training_size', fontsize='large')
    plt.legend(['accuracy', 'f1'], loc='upper left')
    plt.show()
    plt.savefig('hh.png', bbox_inches='tight')


split =10
training_size = []
accuracy = []
precision = []
recall = []
f1 = []
# for i in range(split):
#     training_size.append(i+1000)
#     accuracy.append(i+2)
#     precision.append(i+1)
#     recall.append(i+3)
#     f1.append(i+
res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float),
                       index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
res['accuracy'] = 0.8
res['precision'] = 0.9
res['recall'] = 1
res['duration'] =2
d = res.loc[0,'accuracy']

training_size = [36, 72, 108]
accuracy = [res['accuracy'] , res['accuracy'] , res['accuracy'] ]
f1 = [res['recall'] , res['recall'] , res['recall'] ]
metrics  = {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1,"training_size":training_size}


plot_metrics_matrix()

# plot_trainingsize_metric2(metrics)

plot_trainingsize_metric1(metrics)


print('k')
