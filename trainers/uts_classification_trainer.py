from base.base_trainer import BaseTrain
import os
import time
import keras_contrib
import numpy as np
from utils.uts_classification.utils import save_training_logs
from utils.uts_classification.metric import precision, recall, f1
from utils.uts_classification.tools import AdvancedLearnignRateScheduler


class UtsClassificationTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(UtsClassificationTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        self.init_callbacks()

    def init_callbacks(self):
        if (self.config.model.name == "encoder"):
             import keras
             from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
        else:
            import tensorflow.keras as keras
            from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,'best_model-%s.hdf5'%self.config.callbacks.checkpoint_monitor),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
            )
        )
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001
            )
        )
        self.callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1
            ),
        )
        # 在TCN中使用了tensorflow_addson中的WeightNormalization层，与tensorboard不兼容
        # if (self.config.model.name != "tcn"):
        #     self.callbacks.append(
        #         TensorBoard(
        #             log_dir=self.config.callbacks.tensorboard_log_dir,
        #             write_graph=self.config.callbacks.tensorboard_write_graph,
        #             histogram_freq=1,
        #         )
        #     )
        # if self.config.dataset.name == "ptbdb":
        #     self.callbacks.append(
        #         AdvancedLearnignRateScheduler(monitor='val_main_output_loss', patience=6, verbose=1, mode='auto',
        #                                       decayRatio=0.1),
        #     )

        if ("comet_api_key" in self.config):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
            experiment.disable_mp()
            experiment.log_parameters(self.config["trainer"])
            self.callbacks.append(experiment.get_callback('keras'))

    def train(self):
        if (self.config.model.name == "encoder"):
             import keras
        else:
            import tensorflow.keras as keras
        start_time = time.time()
        if self.config.dataset.name in ['ptbdb','Challeng2018']:

            history = self.model.fit(
                self.data[0], self.data[1],
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose_training,
                batch_size=self.config.trainer.batch_size,
                validation_data = (self.data[2], self.data[3]),
                callbacks=self.callbacks,
            )
        else:
            history = self.model.fit(
                self.data[0], self.data[1],
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose_training,
                batch_size=self.config.trainer.batch_size,
                validation_split=self.config.trainer.validation_split,
                callbacks=self.callbacks,
            )
        self.duration = time.time()-start_time
        self.history = history
        if(self.config.model.name == "encoder"):

            self.best_model = keras.models.load_model(os.path.join(self.config.callbacks.checkpoint_dir,'best_model-%s.hdf5'%self.config.callbacks.checkpoint_monitor),
                            custom_objects={'precision': precision, 'recall': recall,'f1': f1,
                                            'InstanceNormalization': keras_contrib.layers.InstanceNormalization()})
        else:
            self.best_model = keras.models.load_model(os.path.join(self.config.callbacks.checkpoint_dir,'best_model-%s.hdf5'%self.config.callbacks.checkpoint_monitor),
                            custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])
        self.precision.extend(history.history['precision'])
        self.recall.extend(history.history['recall'])
        self.f1.extend(history.history['f1'])
        self.val_precision.extend(history.history['val_precision'])
        self.val_recall.extend(history.history['val_recall'])
        self.val_f1.extend(history.history['val_f1'])

        best_model = save_training_logs(self.config.log_dir,history,lr=False)
        self.best_model_train_loss = best_model.loc[0, 'best_model_train_loss']
        self.best_model_val_loss = best_model.loc[0, 'best_model_val_loss']
        self.best_model_train_acc = best_model.loc[0, 'best_model_train_acc']
        self.best_model_val_acc = best_model.loc[0, 'best_model_val_acc']
        self.best_model_train_precision = best_model.loc[0, 'best_model_train_precision']
        self.best_model_val_precision = best_model.loc[0, 'best_model_val_precision']
        self.best_model_train_recall = best_model.loc[0, 'best_model_train_recall']
        self.best_model_val_recall = best_model.loc[0, 'best_model_val_recall']
        self.best_model_train_f1 = best_model.loc[0, 'best_model_train_f1']
        self.best_model_val_f1 = best_model.loc[0, 'best_model_val_f1']
        self.best_model_learning_rate = best_model.loc[0, 'best_model_learning_rate']
        self.best_model_nb_epoch = best_model.loc[0, 'best_model_nb_epoch']

