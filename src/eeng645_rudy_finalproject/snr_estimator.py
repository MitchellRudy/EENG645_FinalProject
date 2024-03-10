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
    # maxpooling_layer = MaxPooling1D()(model_layers[-1])
    # model_layers.append(maxpooling_layer)
    conv1d_layer_last = Conv1D(num_filters*2,2,strides=2)(model_layers[-1])
    model_layers.append(conv1d_layer_last)
    conv1d_layer_last = Conv1D(num_filters*4,2,strides=2)(model_layers[-1])
    model_layers.append(conv1d_layer_last)
    conv1d_layer_last = Conv1D(num_filters*8,2,strides=2)(model_layers[-1])
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


# # Part 2 - SNR Estimator
#     model_checkpoint_loc = os.path.join(os.getcwd(),"models","model_snr_est_cp.h5")
#     if TRAIN_SNR_MODEL:
#         lr_scheduler_cb = ReduceLROnPlateau(factor = 0.75, patience = 10)
#         checkpoint_model_cb = ModelCheckpoint(model_checkpoint_loc,save_best_only=True)
#         early_stopping_cb = EarlyStopping(patience=10)
#         cbs = [lr_scheduler_cb, checkpoint_model_cb, early_stopping_cb]

#         model_snr = build_snr_estimator()
#         model_snr.fit(
#                     train_batches_pt2, 
#                     epochs=75, 
#                     validation_data=val_batches_pt2,
#                     callbacks=cbs
#                     )
#     else:
#         if os.path.exists(model_checkpoint_loc):
#             loc = os.path.join(os.getcwd(),"models","model_snr_est_cp_val_mse_0626.h5")
#             model_snr = load_model(loc)
#         else:
#             print(f"Couldn't find {model_checkpoint_loc}")

#     snr_preds = model_snr.predict(signals_val_pt2)
#     snr_preds = np.reshape(snr_preds, (snr_preds.shape[0],1))
#     sq_error = np.abs(snr_preds - snrs_val_pt2)**2
#     plt.figure()
#     plt.stem(snrs_val_pt2,sq_error)
#     plt.plot()
#     plt.savefig("snrs vs sq error.png")

#     plt.figure()
#     plt.stem(snrs_val_pt2,snr_preds)
#     plt.plot()
#     plt.savefig("snrs vs preds.png")