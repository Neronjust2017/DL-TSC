from comet_ml import experiment
from data_loader.uts_regression_data_loader import UtsRegressionDataLoader
from models.uts_regression_model import UtsRegressionModel
from trainers.uts_regression_trainer import UtsRegressionModelTrainer
from evaluater.uts_regression_evaluater import UtsRegressionEvaluater
from utils.config import process_config_UtsRegression
from utils.dirs import create_dirs
from utils.utils import get_args
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config_UtsRegression(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = UtsRegressionDataLoader(config)

    print('Create the model.')
    input_shape = [data_loader.w, data_loader.m, config.args.batchSize]
    output_shape = data_loader.m
    model = UtsRegressionModel(config, input_shape, output_shape)

    print('Create the trainer')
    data = [data_loader.get_train_data(), data_loader.get_valid_data(), data_loader.get_test_data()]
    trainer = UtsRegressionModelTrainer(model.model, data, config)

    print('Start training the model.')
    trainer.train()

    print('Create the evaluater.')
    evaluater = UtsRegressionEvaluater(model.model, data[2], config)

    print('Start evaluating the model.')
    evaluater.evluate()


if __name__ == '__main__':
    main()
