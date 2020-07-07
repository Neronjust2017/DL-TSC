from comet_ml import experiment
from data_loader.uts_classification_data_loader import UtsClassificationDataLoader
from models.uts_classification_model import UtsClassificationModel
from trainers.uts_classification_trainer import UtsClassificationTrainer
from evaluater.uts_classification_evaluater import UtsClassificationEvaluater
from utils.config import process_config_UtsClassification
from utils.dirs import create_dirs
from utils.utils import get_args
from comet_ml import Optimizer
import os
import time
from utils.config import get_config_from_json

def process_config_UtsClassification_bayes_optimization(json_file,learning_rate, num_epochs=50, batch_size=16,model_name='fcn'):
    config, _ = get_config_from_json(json_file)

    config.model.name = model_name
    config.model.learning_rate = learning_rate
    config.trainer.num_epochs = num_epochs
    config.trainer.batch_size = batch_size

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

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    args = get_args()
    config, _ = get_config_from_json(args.config)

    bayes_config = {
        "algorithm": "bayes",

        "parameters": {
            # "model": {"type": "categorical", "values": ['cnn','mlp']},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.01},
            # "batch_size": {"type": "integer", "min": 16, "max": 32},
            # "num_epochs": {"type": "integer", "min": 5, "max": 10},
        },
        "spec": {
            "maxCombo": 10,
            "objective": "minimize",
            "metric": "test_f1",
            "minSampleSize": 100,
            "retryAssignLimit": 0,

        },
        "trials": 1,
        "name": "Bayes",
    }
    opt = Optimizer(bayes_config, api_key=config.comet_api_key, project_name=config.exp_name)
    for exp in opt.get_experiments():
        args = get_args()
        # config = process_config_UtsClassification_bayes_optimization(args.config, exp.get_parameter('model'),exp.get_parameter('learning_rate'),
        #                                                              exp.get_parameter('batch_size'), exp.get_parameter('num_epochs'))
        config = process_config_UtsClassification_bayes_optimization(args.config, exp.get_parameter('learning_rate'))
        # except:
        #     print("missing or invalid arguments")
        #     exit(0)

        # create the experiments dirs


        print('Create the data generator.')
        data_loader = UtsClassificationDataLoader(config)

        print('Create the model.')

        model = UtsClassificationModel(config, data_loader.get_inputshape(), data_loader.get_nbclasses())

        print('Create the trainer')
        trainer = UtsClassificationTrainer(model.model, data_loader.get_train_data(), config)

        print('Start training the model.')
        trainer.train()

        # print('Create the evaluater.')
        # evaluater = UtsClassificationEvaluater(trainer.best_model, data_loader.get_test_data(), data_loader.get_nbclasses(),
        #                                        config)
        #
        # print('Start evaluating the model.')
        # evaluater.evluate()

        exp.log_metric("test_f1", trainer.best_model_val_loss)

        print('done')

if __name__ == '__main__':
    main()
