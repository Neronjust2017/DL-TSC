from comet_ml import experiment
from data_loader.uts_classification_data_loader import UtsClassificationDataLoader
from models.uts_classification_model import UtsClassificationModel
from trainers.uts_classification_trainer import UtsClassificationTrainer
from evaluater.uts_classification_evaluater import UtsClassificationEvaluater
from utils.config import process_config_VisTrainingSize
from utils.dirs import create_dirs
from utils.utils import get_args
# from utils.uts_classification.utils import  plot_trainingsize_metric
import pandas as pd
import numpy as  np
import os
import time
import matplotlib.pyplot as plt

def plot_trainingsize_metric(data, file_name):
    plt.figure()
    plt.plot(data["training_size"], data["train_err"])
    plt.plot(data["training_size"], data["val_err"])
    plt.title('learning curve')
    plt.ylabel('Error', fontsize='large')
    plt.xlabel('Training_size', fontsize='large')
    plt.legend(['train_err', 'val_err'], loc='upper right')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close('all')

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    split = 10

    training_size = []
    best_model_train_loss = []
    best_model_val_loss = []
    best_model_train_acc =[]
    best_model_val_acc = []
    best_model_train_precision = []
    best_model_val_precision = []
    best_model_train_recall = []
    best_model_val_recall = []
    best_model_train_f1 = []
    best_model_val_f1 = []
    best_model_learning_rate =[]
    best_model_nb_epoch = []
    best_model_train_err = []
    best_model_val_err = []

    main_dir = ''
    args = get_args()

    for i in range(split):
      config = process_config_VisTrainingSize(args.config,i)
      # except:
      #     print("missing or invalid arguments")
      #     exit(0)

      # create the experiments dirs
      create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir,
                   config.log_dir, config.result_dir])

      print('Create the data generator.')

      data_loader = UtsClassificationDataLoader(config)

      total_train_size = data_loader.get_train_size()

      total_test_size = data_loader.get_test_size()

      print("total_train_size: "+ str(total_train_size))

      print("total_test_size: "+ str(total_test_size))

      print('Create the model.')

      model = UtsClassificationModel(config, data_loader.get_inputshape(), data_loader.get_nbclasses())

      print('Create the trainer')

      train_size = int(total_train_size / split) * (i + 1)

      if i==split-1:
          print("train_size: " + str(total_train_size))

          trainer = UtsClassificationTrainer(model.model, data_loader.get_train_data(), config)

          main_dir = config.main_dir

      else:

         print("train_size: " + str(train_size))

         train_data =data_loader.get_train_data()

         X_train = train_data[0][:train_size,:,:]

         y_train = train_data[1][:train_size,:]

         trainer = UtsClassificationTrainer(model.model,[X_train, y_train], config)

      print('Start training the model.')
      trainer.train()

      best_model_train_loss.append(trainer.best_model_train_loss)
      best_model_val_loss.append(trainer.best_model_val_loss)
      best_model_train_acc.append(trainer.best_model_train_acc)
      best_model_val_acc.append(trainer.best_model_val_acc)
      best_model_train_precision.append(trainer.best_model_train_precision)
      best_model_val_precision.append(trainer.best_model_val_precision)
      best_model_train_recall.append(trainer.best_model_train_recall)
      best_model_val_recall.append(trainer.best_model_val_recall)
      best_model_train_f1.append(trainer.best_model_train_f1)
      best_model_val_f1.append(trainer.best_model_val_f1)
      best_model_learning_rate.append(trainer.best_model_learning_rate)
      best_model_nb_epoch.append(trainer.best_model_nb_epoch)
      best_model_train_err.append(1-trainer.best_model_train_acc)
      best_model_val_err.append(1-trainer.best_model_val_acc)
      training_size.append(train_size)

      print("ss")

    metrics = {"training_size":training_size,"train_err":best_model_train_err,"val_err":best_model_val_err}

    plot_trainingsize_metric(metrics, main_dir + '/vis_overfit_trainingsize.png')

if __name__ == '__main__':
    main()
