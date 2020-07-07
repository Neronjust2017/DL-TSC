import os
import time
from base.base_evaluater import BaseEvaluater
from utils.uts_regression.utils import results2csv, json2csv

class UtsRegressionEvaluater(BaseEvaluater):
    def __init__(self, model, data, config):
        super(UtsRegressionEvaluater,self).__init__(model, data, config)
        self.evluate()

    def evluate(self):
        self.model.load_weights(os.path.join(self.config.callbacks.checkpoint_dir, 'best_model.hdf5'))
        results = self.model.evaluate_generator(self.data, verbose=1)
        csv_dir = self.config.callbacks.base_dir + '/results.csv'
        isExist = os.path.isfile(csv_dir)
        if not isExist:
            results2csv(csv_dir, 'w', ['model', 'time', 'loss', 'mae', 'mse', 'rmse', 'mape', 'msle', 'explained_variance_score'])

        results.insert(0, time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()))
        results.insert(0, self.config.args.model)
        results2csv(csv_dir, 'a', results)

        args_dir = self.config.callbacks.base2_dir + 'args.csv'
        json2csv(args_dir, 'w', self.config.args.toDict())

        print('results: ', results)