from comet_ml import experiment
from data_loader.uts_classification_data_loader import UtsClassificationDataLoader
from models.uts_classification_model import UtsClassificationModel
from trainers.uts_classification_trainer import UtsClassificationTrainer
from evaluater.uts_classification_evaluater import UtsClassificationEvaluater
from utils.config import process_config_UtsClassification, get_config_from_json
from utils.dirs import create_dirs
from utils.utils import get_args
import os
import time
import bayes_opt
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

def process_config_UtsClassification_bayes_optimization(json_file,learning_rate,model_name='fcn', num_epochs=50, batch_size=16 ):
    # print(os.getcwd())
    # print(args.config)

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

if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # print(os.getcwd())
    # os.chdir('../../..')
    # print(os.getcwd())


    def black_box_function(learning_rate):
        # print(os.getcwd())
        # print(args.config)
        args = get_args()

        config = process_config_UtsClassification_bayes_optimization(args.config, learning_rate)

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

        return evaluater.f1

    pbounds = {'learning_rate': (0.0001, 0.01)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        # params={"x": 0.5, "y": 0.7},
        params=[0.001],
        lazy=True,
    )

    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    # By default these will be explored lazily (lazy=True), meaning these points will be evaluated only the next time you call maximize.

    print(optimizer.space.keys)

    optimizer.maximize(init_points=1, n_iter=6)

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)
