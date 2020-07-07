from base.base_evaluater import BaseEvaluater
from utils.uts_classification.utils import save_evaluating_result
import numpy as np
class UtsClassificationEvaluater(BaseEvaluater):
    def __init__(self,model,data,nb_classes,config):
        super(UtsClassificationEvaluater,self).__init__(model,data,config)
        self.nb_classes = nb_classes

    def evluate(self):
        y_pred = self.model.predict(self.data[0])
        loss, accuracy, precision, recall, f1 = self.model.evaluate(self.data[0], self.data[1])
        print('loss:', loss)
        print('accuracy:', accuracy)
        print('precision:', precision)
        print('recall:', recall)
        print('f1:', f1)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = self.data[2]
        if(self.config.model.name == 'tlenet'):
            # get the true predictions of the test set
            tot_increase_num = self.data[3]
            y_predicted = []
            test_num_batch = int(self.data[0].shape[0] / tot_increase_num)
            for i in range(test_num_batch):
                unique_value, sub_ind, correspond_ind, count = np.unique(y_pred, True, True, True)

                idx_max = np.argmax(count)
                predicted_label = unique_value[idx_max]

                y_predicted.append(predicted_label)

            y_pred = np.array(y_predicted)

        cvconfusion,metrics = save_evaluating_result(self.config.result_dir, y_pred, y_true, self.nb_classes)
        self.confusion_matrix = cvconfusion
        self.acc = metrics.loc[0,"Accuracy"]
        self.precision = metrics.loc[0,"Precision(macro)"]
        self.recall = metrics.loc[0,"Recall(macro)"]
        self.f1 = metrics.loc[0,"F1(macro)"]
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []

        for i in range(self.nb_classes):
            self.precision_list.append(metrics.loc[0,'Precison(Cla.' + str(i)+')'])
            self.recall_list.append(metrics.loc[0,'Recall(Cla.' + str(i)+')'])
            self.f1_list.append(metrics.loc[0,'F1(Cla.' + str(i)+')'])


