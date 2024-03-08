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

def build_my_mod_classifier(num_outputs=24, config=None):
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
    # For classification, the abili
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

def build_my_snr_estimator(config=None):
    # Define parameters constant across all model iterations
    input_shape = (1024,2)
    loss_type = "mse"
    model_metric=["mse"]

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
    num_filters = 20
    kernel_size = 2
    input_layer = Input(shape=input_shape)
    num_conv1d_stacks = 1
    num_dense_layers = 5
    num_hidden_neurons = 2**9
    dropout_rate = 0.5

    dilation_rates = 2 ** np.arange(9)
    dilation_rates = dilation_rates.tolist()
    model_layers = []
    model_layers.append(input_layer)
    for _ in range(0,num_conv1d_stacks):
        for idx in range(0,len(dilation_rates)):
            conv1d_layer = Conv1D(num_filters, kernel_size, padding="causal", activation="relu", dilation_rate=dilation_rates[idx])(model_layers[-1])
            model_layers.append(conv1d_layer)
            # batchnorm_layer = BatchNormalization()(model_layers[-1])
            # model_layers.append(batchnorm_layer)

    # conv1d_layer_last = Conv1D(64,1024)(model_layers[-1])
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

def load_expert_model():
    """
    load_expert_model():
    
    Description:
    Loads pre-built model trained to achieve 95% accuracy
    ref: https://www.kaggle.com/code/aleksandrdubrovin/resnet-model-for-radio-signals-classification/notebook#Save-Model-History

    """
    return tf.keras.models.load_model(os.path.join("data","deepsig_io_radioml_2018_01a_dataset","model_full_SNR.h5"))



# def main():
#     data_dir = os.path.join(os.getcwd(),"data","deepsig_io_radioml_2018_01a_dataset")
#     signals_path = os.path.join(data_dir, "signals.npy")
#     labels_path = os.path.join(data_dir, "labels.npy")
#     # Signals Shape (2555904,1024,2)
#     signals = np.load(signals_path)
#     # Labels shape (2555904, 24)
#     labels = np.load(labels_path)

#     return

def preprocess_labels(labels_int, class_labels_keep):
    labels_oh = to_categorical(labels_int)
    labels_oh = labels_oh[:,class_labels_keep]
    # labels_oh = np.reshape(labels_oh, (len(labels_oh),1,len(class_labels_keep)))
    return labels_oh

def main3():
    # Random Seed
    SEED = 1
    # Need to downsample heavily for reasonable runtimes
    # 0.15 -> by a factor of 20
    DOWNSAMPLE_FACTOR = .999
    # 0.075 -> by a factor of 40
    # DOWNSAMPLE_FACTOR = 0.075
    # Train/Val split for models
    VAL_SPLIT = 0.25
    # Model training params
    BUFFER_SIZE = 1000
    BATCH_SIZE = 128
    EPOCHS = 100
    # FLAGS
    TRAIN_MOD_CLASS_MODEL = False
    TRAIN_SNR_TL_MODEL = False
    TRAIN_SNR_MODEL = True
    DO_FIGURES = False

    # Load in the data which was only trimmed by SNR to >=10
    data_storage_dir = os.path.join(os.getcwd(),'data','project')
    signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)

    # Trim down data to "normal" dataset
    class_labels_keep = get_class_labels_normal()
    class_labels_str = get_class_labels_strs(class_labels_keep)
    signals_train_full, labels_int_train_full, snrs_train_full = trim_dataset_by_index(signals_train_full, labels_int_train_full, snrs_train_full, class_labels_keep)
    
    # If using D_F = 0.15:
    # Trim down further to total of ~66,914 samples for all training
    # Est. ~22.3k samples total (~2k samples/class) for each project stage
    # If using D_G = 0.075, halve the above
    idxs = np.arange(len(snrs_train_full))
    # First, break down indices by project parts
    idxs_train_down, idxs_val_down, labels_int_train_down, labels_int_val_down = train_test_split(idxs, labels_int_train_full, random_state=SEED, train_size=DOWNSAMPLE_FACTOR)
    idxs_train_pt1_full, idxs_train_pt23_full, labels_int_train_pt1_full, labels_int_train_pt23_full = train_test_split(idxs_train_down, labels_int_train_down, random_state=SEED, train_size=1/3)
    idxs_train_pt2_full, idxs_train_pt3_full, labels_int_train_pt2_full, labels_int_train_pt3_full = train_test_split(idxs_train_pt23_full, labels_int_train_pt23_full, random_state=SEED, train_size=0.5)
    # Return variables to check distributions are fairly equivalent for each part
    values_pt1, counts_pt1 = np.unique(labels_int_train_pt1_full, return_counts=True)
    values_pt2, counts_pt2 = np.unique(labels_int_train_pt2_full, return_counts=True)
    values_pt3, counts_pt3 = np.unique(labels_int_train_pt3_full, return_counts=True)

    # Now, break down indices by training and validation sets for each part
    idxs_train_pt1, idxs_val_pt1, labels_int_train_pt1, labels_int_val_pt1 = train_test_split(idxs_train_pt1_full, labels_int_train_pt1_full, random_state=SEED, test_size=VAL_SPLIT)
    idxs_train_pt2, idxs_val_pt2, labels_int_train_pt2, labels_int_val_pt2 = train_test_split(idxs_train_pt2_full, labels_int_train_pt2_full, random_state=SEED, test_size=VAL_SPLIT)
    idxs_train_pt3, idxs_val_pt3, labels_int_train_pt3, labels_int_val_pt3 = train_test_split(idxs_train_pt3_full, labels_int_train_pt3_full, random_state=SEED, test_size=VAL_SPLIT)

    # These arrays are the signals and snrs to be used for training/val in a given part
    signals_train_pt1 = signals_train_full[idxs_train_pt1,:,:]
    signals_val_pt1 = signals_train_full[idxs_val_pt1,:,:]

    signals_train_pt2 = signals_train_full[idxs_train_pt2,:,:]
    signals_val_pt2 = signals_train_full[idxs_val_pt2,:,:]

    signals_train_pt3 = signals_train_full[idxs_train_pt3,:,:]
    signals_val_pt3 = signals_train_full[idxs_val_pt3,:,:]
    
    snrs_train_pt1 = snrs_train_full[idxs_train_pt1]
    snrs_val_pt1 = snrs_train_full[idxs_val_pt1]

    snrs_train_pt2 = snrs_train_full[idxs_train_pt2]
    snrs_val_pt2 = snrs_train_full[idxs_val_pt2]

    snrs_train_pt3 = snrs_train_full[idxs_train_pt3]
    snrs_val_pt3 = snrs_train_full[idxs_val_pt3]

    # Convert integer labels BACK to one-hot vectors
    labels_oh_train_pt1 = preprocess_labels(labels_int_train_pt1, class_labels_keep)
    labels_oh_val_pt1 = preprocess_labels(labels_int_val_pt1, class_labels_keep)

    labels_oh_train_pt2 = preprocess_labels(labels_int_train_pt2, class_labels_keep)
    labels_oh_val_pt2 = preprocess_labels(labels_int_val_pt2, class_labels_keep)

    labels_oh_train_pt3 = preprocess_labels(labels_int_train_pt3, class_labels_keep)
    labels_oh_val_pt3 = preprocess_labels(labels_int_val_pt3, class_labels_keep)

    # Build and batch the various dataset objects for model.fit
    # Part 1
    train_dataset_pt1 = Dataset.from_tensor_slices((signals_train_pt1,labels_oh_train_pt1))
    train_batches_pt1 = train_dataset_pt1.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)
    total_train_samples_pt1 = len(train_batches_pt1)*BATCH_SIZE
    val_dataset_pt1 = Dataset.from_tensor_slices((signals_val_pt1,labels_oh_val_pt1))
    val_batches_pt1 = val_dataset_pt1.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)

    signals_test, labels_int_test, snrs_test = trim_dataset_by_index(signals_test, labels_int_test, snrs_test, class_labels_keep)
    labels_test_reshape = preprocess_labels(labels_int_test, class_labels_keep)

    
    lr_scheduler_cb = ReduceLROnPlateau(factor = 0.75, patience = 10)
    model_checkpoint_loc = os.path.join(os.getcwd(),"models","model_mod_class_cp.h5")
    checkpoint_model_cb = ModelCheckpoint(model_checkpoint_loc,save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=10)
    cbs = [lr_scheduler_cb, checkpoint_model_cb, early_stopping_cb]
    if TRAIN_MOD_CLASS_MODEL:
        mod_class_model = build_my_mod_classifier(num_outputs=len(class_labels_keep))

        mod_class_model.fit(
            train_batches_pt1, 
            epochs=EPOCHS, 
            validation_data=val_batches_pt1,
            callbacks=cbs
            )
        mod_class_model = load_model(model_checkpoint_loc)
    else:
        mod_class_model = load_model(os.path.join(os.getcwd(),"models","model_mod_class_val_9776_ds999.h5"))

    if DO_FIGURES:
        # Plotting Confusion Matrix over Pt 1 Validation Data
        y_pred = np.argmax(mod_class_model.predict(signals_val_pt1), axis=1)
        y_true = np.array([np.argmax(x) for x in labels_oh_val_pt1])
        cm_mod_class = confusion_matrix(y_pred=y_pred,y_true=y_true)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm_mod_class, display_labels=class_labels_str)
        cm_disp.plot(cmap=plt.cm.Blues)
        cm_save_loc = os.path.join(os.getcwd(),'figures')
        plt.savefig(os.path.join(cm_save_loc,"cm_mod_class.png"))

    # Part 2
    # MAE
    train_dataset_pt2 = Dataset.from_tensor_slices((signals_train_pt2,snrs_train_pt2))
    train_batches_pt2 = train_dataset_pt2.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)
    total_train_samples_pt2 = len(train_batches_pt2)*BATCH_SIZE
    val_dataset_pt2 = Dataset.from_tensor_slices((signals_val_pt2,snrs_val_pt2))
    val_batches_pt2 = val_dataset_pt2.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)

    # Move onto part 2 with transfer learning attempt
    
    model_checkpoint_loc = os.path.join(os.getcwd(),"models","model_snr_tl_cp.h5")
    if TRAIN_SNR_TL_MODEL:
        # Make the model not trainable
        mod_class_model.trainable = False
        # Want ignore the layer mod_class_model.layers[-1] b/c that was trained for classification
        # Need to do -3 to skip over final dropout layer
        dense_tl_snr1 = Dense(512,name="TL_Dense1",activation="relu")(mod_class_model.layers[-3].output)
        dropout_tl_snr1 = Dropout(0.5,name="TL_Dropout1")(dense_tl_snr1)
        dense_tl_snr2 = Dense(512,name="TL_Dense2",activation="relu")(dropout_tl_snr1)
        dropout_tl_snr2 = Dropout(0.5,name="TL_Dropout2")(dense_tl_snr2)
        dense_tl_snr3 = Dense(512,name="TL_Dense3",activation="relu")(dropout_tl_snr2)
        dropout_tl_snr3 = Dropout(0.5,name="TL_Dropout3")(dense_tl_snr3)
        dense_tl_snr4 = Dense(512,name="TL_Dense4",activation="relu")(dropout_tl_snr3)
        dropout_tl_snr4 = Dropout(0.5,name="TL_Dropout4")(dense_tl_snr4)
        dense_tl_snr5 = Dense(512,name="TL_Dense5",activation="relu")(dropout_tl_snr4)
        dropout_tl_snr5 = Dropout(0.5,name="TL_Dropout5")(dense_tl_snr5)

        dense_output_snr = Dense(1,name="TL_DenseOut")(dropout_tl_snr5)

        model_tl_snr = Model(inputs = mod_class_model.input, outputs=dense_output_snr)

        model_tl_snr.compile(loss="mse", optimizer="adam", metrics="mse")
        model_tl_snr.summary()

        lr_scheduler_cb = ReduceLROnPlateau(factor = 0.75, patience = 10)
        checkpoint_model_cb = ModelCheckpoint(model_checkpoint_loc,save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=10)
        cbs = [lr_scheduler_cb, checkpoint_model_cb, early_stopping_cb]
        
        model_tl_snr.fit(
                train_batches_pt2, 
                epochs=EPOCHS, 
                validation_data=val_batches_pt2,
                callbacks=cbs
                )
    else:
        if os.path.exists(model_checkpoint_loc):
            model_tl_snr = load_model(model_checkpoint_loc)
        else:
            print(f"Couldn't find {model_checkpoint_loc}")

    if DO_FIGURES:
        tl_snr_preds = model_tl_snr.predict(signals_val_pt2)
        sq_error = np.abs(tl_snr_preds - snrs_val_pt2)**2
        plt.figure()
        plt.stem(snrs_val_pt2,sq_error)
        plt.plot()
        plt.savefig("snrs vs sq error (TL).png")

        plt.figure()
        plt.stem(snrs_val_pt2,tl_snr_preds)
        plt.plot()
        plt.savefig("snrs vs preds (TL).png")

    # Part 2 - SNR Estimator
    model_checkpoint_loc = os.path.join(os.getcwd(),"models","model_snr_est_cp.h5")
    if TRAIN_SNR_MODEL:
        lr_scheduler_cb = ReduceLROnPlateau(factor = 0.75, patience = 10)
        checkpoint_model_cb = ModelCheckpoint(model_checkpoint_loc,save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=10)
        cbs = [lr_scheduler_cb, checkpoint_model_cb, early_stopping_cb]

        model_snr = build_my_snr_estimator()
        model_snr.fit(
                    train_batches_pt2, 
                    epochs=50, 
                    validation_data=val_batches_pt2,
                    callbacks=cbs
                    )
    else:
        if os.path.exists(model_checkpoint_loc):
            model_snr = load_model(model_checkpoint_loc)
        else:
            print(f"Couldn't find {model_checkpoint_loc}")

    snr_preds = model_snr.predict(signals_val_pt2)
    snr_preds = np.reshape(snr_preds, (snr_preds.shape[0],1))
    sq_error = np.abs(snr_preds - snrs_val_pt2)**2
    plt.figure()
    plt.stem(snrs_val_pt2,sq_error)
    plt.plot()
    plt.savefig("snrs vs sq error.png")

    plt.figure()
    plt.stem(snrs_val_pt2,snr_preds)
    plt.plot()
    plt.savefig("snrs vs preds.png")
    
    return

if __name__=='__main__':
    main3()
    # t = load_reference_model();
    # t.summary()
    # main()