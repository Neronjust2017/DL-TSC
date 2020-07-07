# -*- coding: UTF-8 -*-
from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils.uts_regression.tools import AdvancedLearnignRateScheduler

class UtsRegressionModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(UtsRegressionModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.mae = []
        self.mse = []
        self.rmse = []
        self.val_loss = []
        self.val_mse = []
        self.val_rmse = []
        self.val_data = data[1]
        self.steps_per_epoch = len(data[0])
        self.validation_steps = len(data[1])
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, 'best_model.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        )

        self.callbacks.append(
            AdvancedLearnignRateScheduler(monitor='val_loss', patience=5, verbose=1, mode='auto', warmup_batches=10,
                                          decayRatio=0.1)
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        # if hasattr(self.config,"comet_api_key"):
        if ("comet_api_key" in self.config):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
            experiment.disable_mp()
            experiment.log_parameters(self.config["args"])
            self.callbacks.append(experiment.get_callback('keras'))

    def train(self):
        history = self.model.fit_generator(
            self.data[0],
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.config.args.epochs,
            validation_data=self.val_data,
            validation_steps=self.validation_steps,
            callbacks=self.callbacks
        )
        self.loss.extend(history.history['loss'])
        self.mae.extend(history.history['mae'])
        self.mse.extend(history.history['mse'])
        self.rmse.extend(history.history['root_mean_square_error'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_mse.extend(history.history['val_mse'])
        self.val_rmse.extend(history.history['val_root_mean_square_error'])