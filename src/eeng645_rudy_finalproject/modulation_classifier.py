import os 

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D


# As point of comparison, use the CNN architecture designed in
# Over the Air Deep Learning Based Radio Signal Classification
# https://arxiv.org/pdf/1712.04578.pdf
def get_paper_cnn_model():
    # REF: Table III. CNN Network LAyout
    # Layer      | Output Dimensions
    #---------------------------
    # Input      | 2 x1024
    # Conv1D     | 64x1024
    # MaxPool    | 64x512
    # Conv1D     | 64x512
    # MaxPool    | 64x256
    # Conv1D     | 64x256
    # MaxPool    | 64x128
    # Conv1D     | 64x128
    # MaxPool    | 64x64
    # Conv1D     | 64x64
    # MaxPool    | 64x32
    # Conv1D     | 64x32
    # MaxPool    | 64x16
    # Conv1D     | 64x16
    # MaxPool    | 64x8
    # FC/Selu    | 128
    # FC/Selu    | 128
    # FC/Softmax | 24
    input_shape = (None,2,1024)
    num_output_classes = 24
    num_filters = 64
    kernel_sizes = [512, 256, 128, 64, 32, 16, 8]
    neurons_selu = 128
    neurons_softmax = num_output_classes
    
    input_layer = Input(shape = input_shape)
    conv1d_1 = Conv1D(num_filters, 1024)(input_layer)
    # 512 kernel_sizes
    maxpool1d_1 = MaxPooling1D(num_filters, kernel_sizes[0])(conv1d_1)
    conv1d_2 = Conv1D(num_filters, kernel_sizes[0])(maxpool1d_1)
    # 256 kernel_sizes
    maxpool1d_2 = MaxPooling1D(num_filters, kernel_sizes[1])(conv1d_2)
    conv1d_3 = Conv1D(num_filters, kernel_sizes[1])(maxpool1d_2)
    # 128 kernel_sizes
    maxpool1d_3 = MaxPooling1D(num_filters, kernel_sizes[2])(conv1d_3)
    conv1d_4 = Conv1D(num_filters, kernel_sizes[2])(maxpool1d_3)
    # 64 kernel_sizes
    maxpool1d_4 = MaxPooling1D(num_filters, kernel_sizes[3])(conv1d_4)
    conv1d_5 = Conv1D(num_filters, kernel_sizes[3])(maxpool1d_4)
    # 32 kernel_sizes
    maxpool1d_5 = MaxPooling1D(num_filters, kernel_sizes[4])(conv1d_5)
    conv1d_6 = Conv1D(num_filters, kernel_sizes[4])(maxpool1d_5)
    # 16 kernel_sizes
    maxpool1d_6 = MaxPooling1D(num_filters, kernel_sizes[5])(conv1d_5)
    conv1d_7 = Conv1D(num_filters, kernel_sizes[5])(maxpool1d_5)
    # 8 kernel_sizes
    maxpool1d_7 = MaxPooling1D(num_filters, kernel_sizes[6])(conv1d_5)

    # Fully connected layer
    dense_selu_1 = Dense(neurons_selu, activation="selu")(maxpool1d_7)
    dense_selu_2 = Dense(neurons_selu, activation="selu")(maxpool1d_7)
    
    # output
    output_layer = Dense(neurons_softmax, activation="softmax")(dense_selu_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile()
    return model


def main():
    data_dir = os.path.join(os.getcwd(),"data","deepsig_io_radioml_2018_01a_dataset")
    signals_path = os.path.join(data_dir, "signals.npy")
    labels_path = os.path.join(data_dir, "labels.npy")
    # Signals Shape (2555904,1024,2)
    signals = np.load(signals_path)
    # Labels shape (2555904, 24)
    labels = np.load(labels_path)

    return

if __name__=='__main__':
    main()