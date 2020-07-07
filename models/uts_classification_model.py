from base.base_model import BaseModel
from utils.uts_classification.metric import f1, recall, precision
from models.classification.fcn import Classifier_FCN
from models.classification.resnet import Classifier_RESNET
from models.classification.cnn import Classifier_CNN
from models.classification.encoder import Classifier_ENCODER
from models.classification.inception import Classifier_INCEPTION
from models.classification.mcdcnn import Classifier_MCDCNN
from models.classification.mlp import Classifier_MLP
from models.classification.resnet_v2 import Classifier_RESNET_V2
from models.classification.tlenet import Classifier_TLENET
from models.classification.resnext import Classifier_RESNEXT
from models.classification.tcn import Classifier_TemporalConvNet
from models.regression.LSTM import LSTM
from models.regression.DeepConvLSTM import DeepConvLSTM
from models.regression.DeepResBiLSTM import DeepResBiLSTM
from models.regression.TCN import TCN


class UtsClassificationModel(BaseModel):
    def __init__(self, config, input_shape, nb_classes):
        super(UtsClassificationModel, self).__init__(config)
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.build_model()
    def build_model(self):
        if self.config.model.name == "inceptiontime":
            self.model = Classifier_INCEPTION(self.input_shape, self.nb_classes, type="inceptiontime").model

        elif self.config.model.name == "inceptiontime_v2":
            self.model = Classifier_INCEPTION(self.input_shape, self.nb_classes, type="inceptiontime_v2").model

        elif self.config.model.name == "resnet":
            self.model = Classifier_RESNET(self.input_shape, self.nb_classes).model

        elif self.config.model.name == "fcn":
            self.model = Classifier_FCN(self.input_shape, self.nb_classes).model

        elif self.config.model.name == "cnn":
            self.model = Classifier_CNN(self.input_shape, self.nb_classes).model

        elif self.config.model.name == "encoder":
            self.model = Classifier_ENCODER(self.input_shape, self.nb_classes).model


        elif self.config.model.name == "mcdcnn":
            self.model = Classifier_MCDCNN(self.input_shape, self.nb_classes).model

        elif self.config.model.name == "mlp":
            self.model = Classifier_MLP(self.input_shape, self.nb_classes).model

        elif self.config.model.name == "resnet_v2":
            self.model = Classifier_RESNET_V2(self.input_shape, self.nb_classes).model

        elif self.config.model.name == "tlenet":
            self.model = Classifier_TLENET().build_model(self.input_shape, self.nb_classes)

        elif self.config.model.name == "resnext":
            self.model = Classifier_RESNEXT(self.input_shape, self.nb_classes).model
        elif self.config.model.name == "tcn":
            self.model = Classifier_TemporalConvNet(self.input_shape, self.nb_classes).model

        self.input_shape = list(self.input_shape)
        self.input_shape.append(self.config.trainer.batch_size)
        if self.config.model.name == "LSTM":
            print('model: LSTM')
            self.model = LSTM(self.input_shape, self.nb_classes).model
        if self.config.model.name == "DeepConvLSTM":
            self.model = DeepConvLSTM(self.input_shape, self.nb_classes).model
        if self.config.model.name == "DeepResBiLSTM":
            self.model = DeepResBiLSTM(self.input_shape, self.nb_classes).model
        if self.config.model.name == "TCN":
            self.model = TCN(self.input_shape, self.nb_classes).model

        optimizer_name = self.config.model.optimizer
        learning_rate = self.config.model.learning_rate

        if self.config.model.name == "encoder":
            import keras
            exec("self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.%s(lr=%s),metrics=['accuracy', precision, recall, f1])"
                 %(optimizer_name, str(learning_rate)))
        else:
            import tensorflow.keras as keras
            optimizer = keras.optimizers.get(optimizer_name)
            optimizer.learning_rate = learning_rate
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                               metrics=['accuracy', precision, recall, f1])
        self.model.summary()