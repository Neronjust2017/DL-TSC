import json
from dotmap import DotMap
import os
import time

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp.name, "logs/")
    config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp.name, "checkpoints/")
    return config

def process_config_UtsClassification(json_file):
    config, _ = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/", time.localtime()),
                                                        config.exp.name, config.dataset.name,
                                                        config.model.name, "%s-%s-%s" % (
                                                        config.model.learning_rate, config.trainer.num_epochs,
                                                        config.trainer.batch_size),
                                                        "tensorboard_logs/")
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

def process_config_VisTrainingSize(json_file, split):
    config, _ = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments",
                                                        config.exp.name, config.dataset.name,
                                                        config.model.name, "vis_overfit_trainingsize", "lr=%s-epoch=%s-batch=%s" % (
                                                            config.model.learning_rate, config.trainer.num_epochs,
                                                            config.trainer.batch_size),"tensorboard_logs","train_split=%s/" % str(split+1))
    config.callbacks.checkpoint_dir = os.path.join("experiments",
                                                   config.exp.name, config.dataset.name,
                                                   config.model.name, "vis_overfit_trainingsize",
                                                   "lr=%s-epoch=%s-batch=%s" % (
                                                       config.model.learning_rate, config.trainer.num_epochs,
                                                       config.trainer.batch_size),"train_split=%s/" % str(split+1),
                                                   "checkpoints/")
    config.main_dir = os.path.join("experiments",
                                  config.exp.name, config.dataset.name,
                                  config.model.name, "vis_overfit_trainingsize",
                                  "lr=%s-epoch=%s-batch=%s" % (
                                      config.model.learning_rate, config.trainer.num_epochs,
                                      config.trainer.batch_size))
    config.log_dir = os.path.join("experiments",
                                  config.exp.name, config.dataset.name,
                                  config.model.name, "vis_overfit_trainingsize",
                                  "lr=%s-epoch=%s-batch=%s" % (
                                      config.model.learning_rate, config.trainer.num_epochs,
                                      config.trainer.batch_size), "train_split=%s/" % str(split + 1),
                                  "training_logs/")
    config.result_dir = os.path.join("experiments",
                                     config.exp.name, config.dataset.name,
                                     config.model.name, "vis_overfit_trainingsize",
                                     "lr=%s-epoch=%s-batch=%s" % (
                                         config.model.learning_rate, config.trainer.num_epochs,
                                         config.trainer.batch_size), "train_split=%s/" % str(split + 1),
                                     "result/")
    return config


def process_config_UtsRegression(json_file):
    config, _ = get_config_from_json(json_file)
    config.callbacks.base_dir = os.path.join("experiments")
    config.callbacks.base2_dir = os.path.join("experiments", config.args.model, time.strftime("%Y-%m-%d-%H-%M-%S/",time.localtime()))
    config.callbacks.tensorboard_log_dir = os.path.join("experiments", config.args.model, time.strftime("%Y-%m-%d-%H-%M-%S/",time.localtime()), "logs/")
    config.callbacks.checkpoint_dir = os.path.join("experiments", config.args.model, time.strftime("%Y-%m-%d-%H-%M-%S/",time.localtime()), "checkpoints/")
    return config