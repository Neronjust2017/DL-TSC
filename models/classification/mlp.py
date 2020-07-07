# MLP
import tensorflow.keras as keras
import numpy as np 
import pandas as pd 
import time

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 


class Classifier_MLP:

	def __init__(self, input_shape, nb_classes):

		self.model = self.build_model(input_shape, nb_classes)

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		# flatten/reshape because when multivariate all should be on the same axis 
		input_layer_flattened = keras.layers.Flatten()(input_layer)

		# keras.layers.Dropout(rate, noise_shape=None, seed=None) 将dropout应用于输入，将输入单元按比例设置为0，防止过拟合
		layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
		layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

		layer_2 = keras.layers.Dropout(0.2)(layer_1)
		layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

		layer_3 = keras.layers.Dropout(0.2)(layer_2)
		layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

		output_layer = keras.layers.Dropout(0.3)(layer_3)
		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		return model

