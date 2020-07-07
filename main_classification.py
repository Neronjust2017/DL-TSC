from comet_ml import experiment
from data_loader.uts_classification_data_loader import UtsClassificationDataLoader
from models.uts_classification_model import UtsClassificationModel
from trainers.uts_classification_trainer import UtsClassificationTrainer
from evaluater.uts_classification_evaluater import UtsClassificationEvaluater
from utils.config import process_config_UtsClassification
from utils.dirs import create_dirs
from utils.utils import get_args
import os
import time
import sys

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    args = get_args()
    config = process_config_UtsClassification(args.config)
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir,
                 config.log_dir, config.result_dir])

    print('Create the data generator.')
    data_loader = UtsClassificationDataLoader(config)

    print("size of dataloader:")
    print(sys.getsizeof(data_loader))

    print('Create the model.')

    model = UtsClassificationModel(config, data_loader.get_inputshape(), data_loader.get_nbclasses())
    print("size of model:")
    print(sys.getsizeof(model))

    print('Create the trainer')
    trainer = UtsClassificationTrainer(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()

    print('Create the evaluater.')
    evaluater = UtsClassificationEvaluater(trainer.best_model, data_loader.get_test_data(), data_loader.get_nbclasses(),
                                           config)

    print('Start evaluating the model.')
    evaluater.evluate()
    print('done')

if __name__ == '__main__':
    main()
