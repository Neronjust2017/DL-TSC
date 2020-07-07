from base.base_model import BaseModel
from utils.uts_classification.metric import f1, recall, precision
from models.classification.fcn import Classifier_FCN
from models.classification.resnet import Classifier_RESNET
from models.classification.cnn import Classifier_CNN
from models.classification.encoder import Classifier_ENCODER
from models.classification.inception import Classifier_INCEPTION
from models.classification.mcdcnn import Classifier_MCDCNN
from models.classification.mlp import Classifier_MLP
from models.classification.resnet_v2_2 import Classifier_RESNET_V2
from models.classification.tlenet import Classifier_TLENET
from models.classification.resnext import Classifier_RESNEXT


class UtsClassificationModel(BaseModel):
    def __init__(self, config, input_shape, nb_classes):
        super(UtsClassificationModel, self).__init__(config)
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.build_model()

    def build_model(self):
        if self.config.model.name == "inceptiontime":
            params = self.config.model.params
            type = params['type']
            nb_filters = params['nb_filters']
            depth = params['depth']
            kernel_size = params['kernel_size']
            self.model = Classifier_INCEPTION(self.input_shape, self.nb_classes, type=type,
                                              nb_filters=nb_filters, use_residual=True, use_bottleneck=True, depth=depth,
                                              kernel_size=kernel_size).model

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
            params = self.config.model.params
            type = params['type']
            convfilt = params['convfilt']
            ksize = params['ksize']
            depth = params['depth']
            drop = params['drop']
            self.model = Classifier_RESNET_V2(self.input_shape, self.nb_classes,type, convfilt, ksize, depth,drop).model

        elif self.config.model.name == "tlenet":
            self.model = Classifier_TLENET().build_model(self.input_shape, self.nb_classes)

        elif self.config.model.name == "resnext":
            self.model = Classifier_RESNEXT(self.input_shape, self.nb_classes).model

        if self.config.model.name == "encoder":
            import keras
        else:
            import tensorflow.keras as keras

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy', precision, recall, f1])