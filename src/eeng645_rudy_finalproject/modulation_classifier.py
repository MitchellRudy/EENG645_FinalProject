# Over the Air Deep Learning Based Radio Signal Classification
# https://arxiv.org/pdf/1712.04578.pdf
import os 

import numpy as np
import matplotlib.pyplot as plt

import ray
from ray import tune
from ray import train
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler #these are aliased in tune (same thing)
from ray.train.tensorflow.keras import ReportCheckpointCallback
from ray.tune.search.optuna import OptunaSearch

import optuna


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard

from data_management import load_train_test_subset, trim_dataset_by_index, get_class_labels_normal, get_class_labels_strs

def build_mod_classifier(num_outputs=24, config=None):
    """
    build_my_mod_classifier()
    Builds my model to classify modulation types

    Inputs:
    config:dictionary - dictionary defining various model parameters

    Outputs:
    model
    """
    # Define parameters constant across all model iterations
    input_shape = (1024,2)
    num_outputs = num_outputs
    if config is None:
        num_filters = 64
        num_kernels = 1024
    # use this loss because labels are converted back to one-hots
    loss_type = "categorical_crossentropy"
    model_metric = [
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall')
    ]
    # Step 5. Initial Model Fitting data shapes
    # input_layer = Input(shape=input_shape)
    # max_pooling = MaxPooling1D(64, 1024)(input_layer)
    # output_layer = Dense(num_outputs, activation="softmax")(max_pooling)
    # model = Model(inputs=input_layer, outputs=output_layer)
    # model.compile(loss=loss_type, 
    #     optimizer=Adam(),
    #     metrics=model_metric)

    # Step 6. Scale up model so it overfits the training data
    # Training for 50 epochs lets model memorize the dataset
    # Ref WAVENET model (pg 691)
    # num_filters = 20
    # kernel_size = 2
    # input_layer = Input(shape=input_shape)
    # num_conv1d_stacks = 1
    # num_dense_layers = 4
    # num_hidden_neurons = 2**11

    # dilation_rates = 2 ** np.arange(9)
    # dilation_rates = dilation_rates.tolist()
    # model_layers = []
    # model_layers.append(input_layer)
    # for _ in range(0,num_conv1d_stacks):
    #     for idx in range(0,len(dilation_rates)):
    #         conv1d_layer = Conv1D(num_filters, kernel_size, padding="causal", activation="relu", dilation_rate=dilation_rates[idx])(model_layers[-1])
    #         model_layers.append(conv1d_layer)

    # # conv1d_layer_last = Conv1D(64,1024)(model_layers[-1])
    # # model_layers.append(conv1d_layer_last)
    # flatten_layer = Flatten()(model_layers[-1])
    # model_layers.append(flatten_layer)

    # # Adopt the "stretch pants" approach (book 425/1150)
    # for _ in range(0,num_dense_layers):
    #     dense_layer = Dense(num_hidden_neurons, activation="relu")(model_layers[-1])
    #     model_layers.append(dense_layer)


    # output_layer = Dense(num_outputs, activation="softmax")(model_layers[-1])
    # model = Model(inputs=input_layer, outputs=output_layer)
    # model.compile(loss=loss_type, 
    #     optimizer=Adam(),
    #     metrics=model_metric)

    num_filters = 64
    kernel_size = 2
    input_layer = Input(shape=input_shape)
    num_dense_layers = 4
    num_conv1d_layers = 7
    num_hidden_neurons = 512
    dropout_rate = 0.5

    # Step 7. Scaling up
    # At this point, the model morphs into a form nearly identical to the paper's model
    model_layers = []
    model_layers.append(input_layer)
    for idx in range(0,num_conv1d_layers):
        conv1d_layer = Conv1D(num_filters, 1,padding='valid', activation="relu")(model_layers[-1])
        model_layers.append(conv1d_layer)
        maxpool1d_layer = MaxPooling1D(pool_size=2,strides=None, padding='valid')(model_layers[-1])
        model_layers.append(maxpool1d_layer)
        # batchnorm_layer = BatchNormalization()(model_layers[-1])
        # model_layers.append(batchnorm_layer)

    flatten_layer = Flatten()(model_layers[-1])
    model_layers.append(flatten_layer)

    # Adopt the "stretch pants" approach (book 425/1150)
    for _ in range(0,num_dense_layers):
        dense_layer = Dense(num_hidden_neurons, activation="relu")(model_layers[-1])
        model_layers.append(dense_layer)
        dropout_layer = Dropout(dropout_rate)(model_layers[-1])
        model_layers.append(dropout_layer)


    output_layer = Dense(num_outputs, activation="softmax")(model_layers[-1])
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss_type, 
        optimizer=Adam(),
        metrics=model_metric)

    model.summary()
    return model
