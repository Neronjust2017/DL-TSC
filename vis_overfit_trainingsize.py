from comet_ml import experiment
from data_loader.uts_classification_data_loader import UtsClassificationDataLoader
from models.uts_classification_model import UtsClassificationModel
from trainers.uts_classification_trainer import UtsClassificationTrainer
from evaluater.uts_classification_evaluater import UtsClassificationEvaluater
from utils.config import process_config_VisOverfit
from utils.dirs import create_dirs
from utils.utils import get_args
# from utils.uts_classification.utils import  plot_trainingsize_metric
import pandas as pd
import numpy as  np
import os
import time

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    split = 10

    training_size = []
    accuracy = []
    precision = []
    recall = []
    f1 = []

    main_dir = ''
    args = get_args()

    for i in range(split):
      config = process_config_VisOverfit(args.config,i)
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

      print('Create the evaluater.')
      evaluater = UtsClassificationEvaluater(trainer.best_model, data_loader.get_test_data(), data_loader.get_nbclasses(),
                                                   config)

      print('Start evaluating the model.')
      evaluater.evluate()

      training_size.append(train_size)

      accuracy.append(evaluater.acc)
      precision.append(evaluater.precision)
      recall.append(evaluater.recall)
      f1.append(evaluater.f1)
      print("ss")

    metrics = {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1,"training_size":training_size}

    plot_trainingsize_metric(metrics, main_dir + 'vis_overfit_trainingsize.png')

if __name__ == '__main__':
    main()
