import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
###################################################################
### Callback method for reducing learning rate during training  ###
###################################################################
class AdvancedLearnignRateScheduler(Callback):

    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto', decayRatio=0.1, warmup_batches=-1, init_lr=0.001):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio
        self.warmup_batches = warmup_batches
        self.batch_count = 0
        self.init_lr = init_lr

        if mode not in ['auto', 'min', 'max']:
            # warnings.warn('Mode %s is unknown, '
            #               'fallback to auto mode.'
            #               % (self.mode), RuntimeWarning)
            print('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            # warnings.warn('AdvancedLearnignRateScheduler'
            #               ' requires %s available!' %
            #               (self.monitor), RuntimeWarning)
            print('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                current_lr = K.get_value(self.model.optimizer.lr)
                new_lr = current_lr * self.decayRatio
                self.init_lr = self.init_lr * self.decayRatio
                K.set_value(self.model.optimizer.lr, new_lr)
                self.wait = 0
            self.wait += 1

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            # if self.verbose > 0:
            #     print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
            #           'rate to %s.' % (self.batch_count + 1, lr))

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1

class AdvancedLearnignRateScheduler_WeightUpdate(Callback):

    def __init__(self, weight_path, monitor='val_loss', patience=0, verbose=0, mode='auto', decayRatio=0.1, warmup_batches=-1, init_lr=None):
        super(Callback, self).__init__()
        self.weight_path = weight_path
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio
        self.warmup_batches = warmup_batches
        self.batch_count = 0
        self.init_lr = init_lr

        if mode not in ['auto', 'min', 'max']:
            # warnings.warn('Mode %s is unknown, '
            #               'fallback to auto mode.'
            #               % (self.mode), RuntimeWarning)
            print('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            # warnings.warn('AdvancedLearnignRateScheduler'
            #               ' requires %s available!' %
            #               (self.monitor), RuntimeWarning)
            print('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                current_lr = K.get_value(self.model.optimizer.lr)
                new_lr = current_lr * self.decayRatio
                if self.init_lr is not None:
                    self.init_lr = self.init_lr * self.decayRatio
                K.set_value(self.model.optimizer.lr, new_lr)
                self.wait = 0
                self.model.load_weights(self.weight_path)
                print('best model weight loaded')
            self.wait += 1

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            if self.init_lr is not None:
                lr = self.batch_count * self.init_lr / self.warmup_batches
                K.set_value(self.model.optimizer.lr, lr)
            # if self.verbose > 0:
            #     print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
            #           'rate to %s.' % (self.batch_count + 1, lr))

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1