import os 

import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from data_management import load_train_test_subset, trim_dataset_by_index, get_class_labels_normal, get_class_labels_strs

# Instructor provided function
def plot_confusion_matrix(cm, classes,fig_path, fig_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{fig_path}/{title}')

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
    num_conv1d_stacks = 1
    num_dense_layers = 4
    num_conv1d_layers = 8
    num_hidden_neurons = 256
    dropout_rate = 0.5

    dilation_rates = 2 ** np.arange(7)
    dilation_rates = dilation_rates.tolist()
    model_layers = []
    model_layers.append(input_layer)
    for _ in range(0,num_conv1d_stacks):
        for idx in range(0,num_conv1d_layers):
            conv1d_layer = Conv1D(num_filters, kernel_sizes[idx],padding='valid', activation="relu")(model_layers[-1])
            model_layers.append(conv1d_layer)
            batchnorm_layer = BatchNormalization()(model_layers[-1])
            model_layers.append(batchnorm_layer)

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


    output_layer = Dense(num_outputs, activation="softmax")(model_layers[-1])
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

def main2():
    data_storage_dir = os.path.join(os.getcwd(),'data','project')
    signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)
    t = load_expert_model()
    predvals = t.predict(signals_train_full[0:32,:,:])
    preds = np.zeros(32)
    for idx in range(0,32):
        preds[idx] = np.argmax(predvals[idx,:])
    true_labels = labels_int_train_full[0:32]
    error = np.sum(abs(preds!=true_labels))
    perc_error = error/32*100
    print(f"Percent Error {perc_error}")
    return

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
    DOWNSAMPLE_FACTOR = 0.15
    # 0.075 -> by a factor of 40
    # DOWNSAMPLE_FACTOR = 0.075
    # Train/Val split for models
    VAL_SPLIT = 0.25
    # Model training params
    BUFFER_SIZE = 1000
    BATCH_SIZE = 128
    EPOCHS = 50
    # FLAGS
    TRAIN_MOD_CLASS_MODEL = False

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
    train_dataset_pt1 = Dataset.from_tensor_slices((signals_train_pt1,labels_oh_train_pt1))
    train_batches_pt1 = train_dataset_pt1.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)
    total_train_samples_pt1 = len(train_batches_pt1)*BATCH_SIZE
    val_dataset_pt1 = Dataset.from_tensor_slices((signals_val_pt1,labels_oh_val_pt1))
    val_batches_pt1 = val_dataset_pt1.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)

    signals_test, labels_int_test, snrs_test = trim_dataset_by_index(signals_test, labels_int_test, snrs_test, class_labels_keep)
    labels_test_reshape = preprocess_labels(labels_int_test, class_labels_keep)



    if TRAIN_MOD_CLASS_MODEL:
        mod_class_model = build_my_mod_classifier(num_outputs=len(class_labels_keep))

        mod_class_model.fit(train_batches_pt1, epochs=EPOCHS, validation_data=val_batches_pt1)

        mod_class_model.save(os.path.join(os.getcwd(),"models","mod_class.h5"))
    else:
        mod_class_model = load_model(os.path.join(os.getcwd(),"models","mod_class_val_acc81.h5"))
    y_pred = np.argmax(mod_class_model.predict(signals_val_pt1), axis=1)
    y_true = np.array([np.argmax(x) for x in labels_oh_val_pt1])
    cm_mod_class = confusion_matrix(y_pred=y_pred,y_true=y_true)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm_mod_class, display_labels=class_labels_str)
    cm_disp.plot(cmap=plt.cm.Blues)
    cm_save_loc = os.path.join(os.getcwd(),'figures')
    plt.savefig(os.path.join(cm_save_loc,"cm_mod_class.png"))

    return

if __name__=='__main__':
    main3()
    # t = load_reference_model();
    # t.summary()
    # main()