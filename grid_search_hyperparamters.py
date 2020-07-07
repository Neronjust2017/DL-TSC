from comet_ml import experiment
from data_loader.uts_classification_data_loader import UtsClassificationDataLoader
from models.uts_classification_model import UtsClassificationModel
from trainers.uts_classification_trainer import UtsClassificationTrainer
from evaluater.uts_classification_evaluater import UtsClassificationEvaluater
from utils.config import process_config_UtsClassification
from utils.dirs import create_dirs
from utils.utils import get_args
from keras import backend as K
# from utils.uts_classification.utils import  plot_trainingsize_metric
import pandas as pd
import numpy as  np
import os
import time
import matplotlib.pyplot as plt
from utils.config import get_config_from_json
def process_config_UtsClassification_grid_search_hyperparamters(json_file,model_name, learning_rate):
    config, _ = get_config_from_json(json_file)

    config.model.name = model_name
    config.model.learning_rate = learning_rate

    config.callbacks.tensorboard_log_dir = os.path.join("experiments",time.strftime("%Y-%m-%d/", time.localtime()),
                                                        config.exp.name, config.dataset.name,
                                                        config.model.name, "tensorboard_logs",
                                                        "lr=%s,epoch=%s,batch=%s" % (
                                                            config.model.learning_rate, config.trainer.num_epochs,
                                                            config.trainer.batch_size)
                                                       )
    config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/", time.localtime()),
                                                   config.exp.name, config.dataset.name,
                                                   config.model.name, "%s-%s-%s" % (
                                                   config.model.learning_rate, config.trainer.num_epochs,
                                                   config.trainer.batch_size),
                                                   "checkpoints/")
    config.log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/", time.localtime()),
                                            config.exp.name, config.dataset.name,
                                            config.model.name, "%s-%s-%s" % (
                                            config.model.learning_rate, config.trainer.num_epochs,
                                            config.trainer.batch_size),
                                            "training_logs/")
    config.result_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/", time.localtime()),
                                               config.exp.name, config.dataset.name,
                                               config.model.name, "%s-%s-%s" % (
                                                   config.model.learning_rate, config.trainer.num_epochs,
                                                   config.trainer.batch_size),
                                               "result/")
    return config

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for model_name in ['fcn','resnet_v2','cnn']:
        for learning_rate in [0.001, 0.0005, 0.0001]:
            args = get_args()
            config = process_config_UtsClassification_grid_search_hyperparamters(args.config, model_name, learning_rate)
            # except:
            #     print("missing or invalid arguments")
            #     exit(0)

            # create the experiments dirs


            create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir,
                         config.log_dir, config.result_dir])

            print('Create the data generator.')
            data_loader = UtsClassificationDataLoader(config)

            print('Create the model.')

            model = UtsClassificationModel(config, data_loader.get_inputshape(), data_loader.get_nbclasses())

            print('Create the trainer')
            trainer = UtsClassificationTrainer(model.model, data_loader.get_train_data(), config)

            print('Start training the model.')
            trainer.train()

            print('Create the evaluater.')
            evaluater = UtsClassificationEvaluater(trainer.best_model, data_loader.get_test_data(),
                                                   data_loader.get_nbclasses(),
                                                   config)

            print('Start evaluating the model.')
            evaluater.evluate()
            print('done')

            K.clear_session()


if __name__ == '__main__':
    main()
