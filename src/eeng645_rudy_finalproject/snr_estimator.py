import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras import Model

def build_snr_estimator(config=None):
    # Define parameters constant across all model iterations
    input_shape = (1024,2)
    loss_type = "mse"
    model_metric=["mae"]

    # Step 5. Build Initial Model
    # input_layer = Input(shape=input_shape)
    # max_pooling = MaxPooling1D(64, 1024)(input_layer)
    # output_layer = Dense(1)(max_pooling)
    # model = Model(inputs=input_layer, outputs=output_layer)
    # model.compile(loss=loss_type, 
    #     optimizer=Adam(),
    #     metrics=model_metric)

    # Step 6. Overfit the Training Data
    # input_layer = Input(shape=input_shape)
    # num_filters = 20
    # kernel_size = 2
    # input_layer = Input(shape=input_shape)
    # num_conv1d_stacks = 1
    # num_dense_layers = 4
    # num_hidden_neurons = 2**8

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

    # output_layer = Dense(1)(model_layers[-1])
    # model = Model(inputs=input_layer, outputs=output_layer)
    # model.compile(loss=loss_type, 
    #     optimizer=Adam(),
    #     metrics=model_metric)


    # Step 7. Regularize the Model
    input_layer = Input(shape=input_shape)
    num_filters = 32
    kernel_size = 2
    input_layer = Input(shape=input_shape)
    num_conv1d_stacks = 1
    num_dense_layers = 4
    num_hidden_neurons = 2**7
    dropout_rate = 0.25

    dilation_rates = 2 ** np.arange(8)
    dilation_rates = dilation_rates.tolist()
    model_layers = []
    model_layers.append(input_layer)
    for _ in range(0,num_conv1d_stacks):
        for idx in range(0,len(dilation_rates)):
            conv1d_layer = Conv1D(num_filters, kernel_size, padding="causal", activation="relu", dilation_rate=dilation_rates[idx])(model_layers[-1])
            model_layers.append(conv1d_layer)

    # downscale number of parameters
    conv1d_layer_last = Conv1D(num_filters*2,2,strides=2)(model_layers[-1])
    model_layers.append(conv1d_layer_last)

    conv1d_layer_last = Conv1D(num_filters*4,2,strides=2)(model_layers[-1])
    model_layers.append(conv1d_layer_last)

    maxpooling_layer = MaxPooling1D(pool_size=256,strides=None,padding='valid')(model_layers[-1])
    model_layers.append(maxpooling_layer)

    # conv1d_layer_last = Conv1D(num_filters,2,strides=2)(model_layers[-1])
    # model_layers.append(conv1d_layer_last)
    flatten_layer = Flatten()(model_layers[-1])
    model_layers.append(flatten_layer)

    # Adopt the "stretch pants" approach (book 425/1150)
    for _ in range(0,num_dense_layers):
        dense_layer = Dense(num_hidden_neurons, activation="relu")(model_layers[-1])
        model_layers.append(dense_layer)
        dropout_layer = Dropout(dropout_rate)(model_layers[-1])
        model_layers.append(dropout_layer)

    output_layer = Dense(1)(model_layers[-1])
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss_type, 
        optimizer=Adam(),
        metrics=model_metric)

    model.summary()
    return model
