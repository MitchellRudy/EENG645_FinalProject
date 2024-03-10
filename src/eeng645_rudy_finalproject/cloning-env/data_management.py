# EENG 645A - Final Project
# data_management.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import time

def save_train_test_subset(
    base_dir: str, 
    snr_thresh = 10, 
    seed = 1, 
    datakeep_percentage = 0.50,
    test_percentage = 0.10,
    ):
    """
    def save_train_test_subset(
        base_dir: str, 
        snr_thresh = 10, 
        seed = 1, 
        datakeep_percentage = 0.50,
        test_percentage = 0.10,
    ):

    Description:
    Load in the full deepsig_io_radioml_2018_01a_dataset dataset, then trim it down based on the snr_thresh value.
    Split remaining dataset into a training and test dataset via train_test_split, based on the datakeep_percentage.
    The training set is what will be used across the various project stages. The returned "test" set shall be discarded.
    This is done due to memory requirements
    TODO: TF Datasets?

    Inputs:
    base_dir: str - Path to directory where project's numpy arrays shall be saved
    snr_thresh=10 - Lower bound on snr range kept and saved into new arrays for project
    seed=1 - seed to control train/test splitting randomness
    datakeep_percentage=0.50 - percentage of dataset to keep in the training subset 
    test_percentage=0.10 - percentage of each part's data to split off into test group

    Outputs:
    data_package: str - f'snr{snr_thresh}_keep{int(datakeep_percentage*100)}_test{int(test_percentage*100)}_seed{seed}'
    """
    ##############################
    ##### Dataset Label Info #####
    ##############################

    # By default, the labels are ONE-HOT ENCODED for the following classes
    # Index: Modulation Type
    # 0 :'32PSK', (32 Phase-Shift Keying)
    # 1: '16APSK', (16 Amplitude/Phase Shift Keying)
    # 2: '32QAM', (32 Quadrature Amplitude Modulation)
    # 3: 'FM', (Frequency Modulation)
    # 4: 'GMSK', (Gaussian minimum-shift keying)
    # 5: '32APSK', (32 Amplitude/Phase Shift Keying)
    # 6: 'OQPSK', (Offset Quadrature Phase Shift Keying)
    # 7: '8ASK', (8 Amplitude Shift Keying)
    # 8: 'BPSK', (Binary Phase Shift Keying)
    # 9: '8PSK', (8 Phase Shift Keying)
    # 10: 'AM-SSB-SC', (Amplitude Modulation Single Sideband, Suppressed Carrier)
    # 11: '4ASK', (4 Amplitude Shift Keying)
    # 12: '16PSK', (16 Phase Shift Keying)
    # 13: '64APSK', (64 Amplitude Phase Shift Keying)
    # 14: '128QAM', (128 Quadrature Amplitude Modulation)
    # 15: '128APSK', (128 Amplitude Phase Shift Keying)
    # 16: 'AM-DSB-SC', (Amplitude Modulation, Double Sideband, Suppressed Carrer)
    # 17: 'AM-SSB-WC', (Amplitude Modulation, Single Sideband, With Carrier)
    # 18: '64QAM', (64 Quadrature Amplitude Modulation)
    # 19: 'QPSK', (Quadrature Phase Shift Keying)
    # 20: '256QAM', (256 Quadrature Amplitude Modulation)
    # 21: 'AM-DSB-WC', (Amplitude Modulation, Double Sideband, With Carrier)
    # 22: 'OOK', (On-Off Keying)
    # 23: '16QAM' (16  Quadrature Amplitude Modulation)

    # SNR range: -20:2:30
    
    ##########################
    ##### DIRECTORY PREP #####
    ##########################

    # Add subdirectory to indicate the generation parameters
    data_package = f'snr{snr_thresh}_keep{int(datakeep_percentage*100)}_test{int(test_percentage*100)}_seed{seed}'
    base_dir = os.path.join(base_dir,data_package)
    # Folder to hold data prior to splitting into each part
    full_trimmed_dataset_dir = os.path.join(base_dir, "full_data_used")
    # The folders for each part will have both training and testing
    pt1_data_dir = os.path.join(base_dir, "pt1_data")
    pt2_data_dir = os.path.join(base_dir, "pt2_data")
    pt3_data_dir = os.path.join(base_dir, "pt3_data")

    # Make directories for data
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        os.mkdir(full_trimmed_dataset_dir)
        os.mkdir(pt1_data_dir)
        os.mkdir(pt2_data_dir)
        os.mkdir(pt3_data_dir)


    # Save locations for the total set of data used in this project
    full_signals_saveloc = os.path.join(full_trimmed_dataset_dir, 'full_signals.npy')
    full_labels_saveloc = os.path.join(full_trimmed_dataset_dir, 'full_labels.npy')
    full_snrs_saveloc = os.path.join(full_trimmed_dataset_dir, 'full_snrs.npy')
    
    # Save locations for the Part 1 data
    pt1_signals_train_saveloc = os.path.join(pt1_data_dir, 'signals_train.npy')
    pt1_signals_test_saveloc = os.path.join(pt1_data_dir, 'signals_test.npy')

    pt1_labels_train_saveloc = os.path.join(pt1_data_dir, 'labels_train.npy')
    pt1_labels_test_saveloc = os.path.join(pt1_data_dir, 'labels_test.npy')

    pt1_snrs_train_saveloc = os.path.join(pt1_data_dir, 'snrs_train.npy')
    pt1_snrs_test_saveloc = os.path.join(pt1_data_dir, 'snrs_test.npy')

    # Save locations for the Part 2 data
    pt2_signals_train_saveloc = os.path.join(pt2_data_dir, 'signals_train.npy')
    pt2_signals_test_saveloc = os.path.join(pt2_data_dir, 'signals_test.npy')
    
    pt2_labels_train_saveloc = os.path.join(pt2_data_dir, 'labels_train.npy')
    pt2_labels_test_saveloc = os.path.join(pt2_data_dir, 'labels_test.npy')

    pt2_snrs_train_saveloc = os.path.join(pt2_data_dir, 'snrs_train.npy')
    pt2_snrs_test_saveloc = os.path.join(pt2_data_dir, 'snrs_test.npy')    
    
    # Save locations for the Part 3 data
    pt3_signals_train_saveloc = os.path.join(pt3_data_dir, 'signals_train.npy')
    pt3_signals_test_saveloc = os.path.join(pt3_data_dir, 'signals_test.npy')
    
    pt3_labels_train_saveloc = os.path.join(pt3_data_dir, 'labels_train.npy')
    pt3_labels_test_saveloc = os.path.join(pt3_data_dir, 'labels_test.npy')

    pt3_snrs_train_saveloc = os.path.join(pt3_data_dir, 'snrs_train.npy')
    pt3_snrs_test_saveloc = os.path.join(pt3_data_dir, 'snrs_test.npy')

    #########################
    ##### Data Trimming #####
    #########################
    
    # Load in the FULL signals and labels files
    signals = np.load("data/deepsig_io_radioml_2018_01a_dataset/signals.npy")
    snrs = np.load("data/deepsig_io_radioml_2018_01a_dataset/snrs.npy")
    labels = np.load("data/deepsig_io_radioml_2018_01a_dataset/labels.npy")
    print(f"Loaded in signals.npy, snrs.npy, and labels.npy")

    # Convert via np.argmax(labels[row]) for easier filtering later
    labels_int = np.asarray([np.argmax(labels[i]) for i in range(0, len(labels))])
    print("Converted labels from one-hot encoding to unique numbers")
    
    # Used masked array to filter by SNR (save memory)
    # https://numpy.org/doc/stable/reference/maskedarray.generic.html
    # https://numpy.org/doc/stable/reference/generated/numpy.ma.masked_where.html#numpy.ma.masked_where
    snr_masked_array = np.ma.masked_where(snrs >= snr_thresh, snrs)
    snr_mask = snr_masked_array.mask
    # Reshape to ensure not an array of single element arrays
    snr_mask = np.reshape(snr_mask,(len(labels),))

    # Trim down the full datasets based on the SNR mask
    signals = signals[snr_mask]
    snrs = snrs[snr_mask]
    labels_int = labels_int[snr_mask]

    # Perform train test split on indices
    # https://stackoverflow.com/questions/31467487/memory-efficient-way-to-split-large-numpy-array-into-train-and-test
    indices = np.arange(len(signals))

    # Split to get the full set of data used across this project
    if int(datakeep_percentage) != 1:
        indices_full, _, labels_int_full, _ = train_test_split(indices, labels_int, random_state=seed, train_size=datakeep_percentage)
        signals_full = signals[indices_full]
        snrs_full = snrs[indices_full]
    else:
        signals_full = signals
        labels_int_full = labels_int
        snrs_full = snrs
    
    # Save it    
    np.save(full_signals_saveloc, signals_full)
    np.save(full_labels_saveloc, labels_int_full)
    np.save(full_snrs_saveloc, snrs_full)

    print(f"Saved full set of data for this project to: {full_trimmed_dataset_dir}")

    ###############################
    ##### Split Data By Parts #####
    ###############################

    # Now, need to split the remaining data into approximately equal thirds
    indices = np.arange(len(snrs_full))
    # Do this by first doing a train_test_split call with train_size=1/3 to get (1) -> (1/3, 2/3)
    indices_pt1, indices_pt23, labels_int_pt1, labels_int_pt23 = train_test_split(indices, labels_int_full, random_state=seed, train_size=1/3)
    # Then, take the "_pt23" output from train_test_split and pass it into a second train_test_split call with
    #   train_size = 0.5 to get (2/3) -> (1/3, 1/3)
    indices_pt2, indices_pt3, labels_int_pt2, labels_int_pt3 = train_test_split(indices_pt23, labels_int_pt23, random_state=seed, train_size=0.5)

    # Now that indices are separated by parts, separate them into the training/val indices and testing indices
    # Part 1
    indices_pt1_trainval, indices_pt1_test, labels_int_pt1_trainval, labels_int_pt1_test = train_test_split(indices_pt1, labels_int_pt1, random_state=seed, test_size=test_percentage)
    # Part 2
    indices_pt2_trainval, indices_pt2_test, labels_int_pt2_trainval, labels_int_pt2_test = train_test_split(indices_pt2, labels_int_pt2, random_state=seed, test_size=test_percentage)
    # Part 3
    indices_pt3_trainval, indices_pt3_test, labels_int_pt3_trainval, labels_int_pt3_test = train_test_split(indices_pt3, labels_int_pt3, random_state=seed, test_size=test_percentage)

    # Use the indices to slice the full data array, then save the arrays
    # Part 1
    signals_pt1_trainval = signals_full[indices_pt1_trainval]
    snrs_pt1_trainval = snrs_full[indices_pt1_trainval]    

    np.save(pt1_signals_train_saveloc, signals_pt1_trainval)
    np.save(pt1_labels_train_saveloc, labels_int_pt1_trainval)
    np.save(pt1_snrs_train_saveloc, snrs_pt1_trainval)

    signals_pt1_test = signals_full[indices_pt1_test]
    snrs_pt1_test = snrs_full[indices_pt1_test]
    
    np.save(pt1_signals_test_saveloc, signals_pt1_test)
    np.save(pt1_labels_test_saveloc, labels_int_pt1_test)
    np.save(pt1_snrs_test_saveloc, snrs_pt1_test)

    print(f"Saved part 1 datasets to: {pt1_data_dir}")
    
    # Part 2
    signals_pt2_trainval = signals_full[indices_pt2_trainval]
    snrs_pt2_trainval = snrs_full[indices_pt2_trainval]    

    np.save(pt2_signals_train_saveloc, signals_pt2_trainval)
    np.save(pt2_labels_train_saveloc, labels_int_pt2_trainval)
    np.save(pt2_snrs_train_saveloc, snrs_pt2_trainval)

    signals_pt2_test = signals_full[indices_pt2_test]
    snrs_pt2_test = snrs_full[indices_pt2_test]
    
    np.save(pt2_signals_test_saveloc, signals_pt2_test)
    np.save(pt2_labels_test_saveloc, labels_int_pt2_test)
    np.save(pt2_snrs_test_saveloc, snrs_pt2_test)
    
    print(f"Saved part 2 datasets to: {pt2_data_dir}")
    
    # Part 3
    signals_pt3_trainval = signals_full[indices_pt3_trainval]
    snrs_pt3_trainval = snrs_full[indices_pt3_trainval]    

    np.save(pt3_signals_train_saveloc, signals_pt3_trainval)
    np.save(pt3_labels_train_saveloc, labels_int_pt3_trainval)
    np.save(pt3_snrs_train_saveloc, snrs_pt3_trainval)

    signals_pt3_test = signals_full[indices_pt3_test]
    snrs_pt3_test = snrs_full[indices_pt3_test]
    
    np.save(pt3_signals_test_saveloc, signals_pt3_test)
    np.save(pt3_labels_test_saveloc, labels_int_pt3_test)
    np.save(pt3_snrs_test_saveloc, snrs_pt3_test)

    print(f"Saved part 3 datasets to: {pt3_data_dir}")

    return data_package

def load_train_test_subset(base_dir):
    """
    load_train_test_subset(base_dir):
    
    Description:
    Load the numpy arrays used for training and testing the models.
    To create a subset, use "save_train_test_subset"

    Inputs:
    base_dir: str - Path to directory where numpy arrays are stored

    Outputs:
    signals_train: nparray - Numpy array containing IQ arrays for train+validation signals
    labels_int_train: nparray - Numpy array containing classification labels for train+validation signals
    snrs_train_full: nparray - Numpy array containing snrs for train+validation signals
    signals_test: nparray - Numpy array containing IQ arrays for test signals
    labels_int_test: nparray - Numpy array containing classification labels for test signals
    snrs_test: nparray - Numpy array containing snrs for test signals
    """

    if not (os.path.exists(base_dir)):
        print(f"{base_dir} not found. Returning None")
        return None, None, None, None, None, None

    print(f"Searching {base_dir} for data...")
    # Load in the training data, return None if there is an issue
    try:
        signals_train = np.load(os.path.join(base_dir,'signals_train.npy'))
        print("Loaded training signals.")
    except:
        print("Error loading training signals. Returning None.")
        signals_train = None

    try:
        labels_int_train = np.load(os.path.join(base_dir,'labels_train.npy'))
        print("Loaded training labels.")
    except:
        print("Error loading training labels. Returning None.")
        labels_int_train = None

    try:
        snrs_train = np.load(os.path.join(base_dir,'snrs_train.npy'))
        print("Loaded training snrs.")
    except:
        print("Error loading training snrs. Returning None.")
        snrs_train = None

    # Load in the test data
    try:
        signals_test= np.load(os.path.join(base_dir,'signals_test.npy'))
        print("Loaded test signals.")
    except:
        print("Error loading test signals. Returning None.")
        signals_test = None

    try:
        labels_int_test = np.load(os.path.join(base_dir,'labels_test.npy'))
        print("Loaded test labels.")
    except:
        print("Error loading test labels. Returning None.")
        labels_int_test = None

    try:
        snrs_test = np.load(os.path.join(base_dir,'snrs_test.npy'))
        print("Loaded test snrs.")
    except:
        print("Error loading test snrs. Returning None.")
        snrs_test = None

    return signals_train, labels_int_train, snrs_train, signals_test, labels_int_test, snrs_test

def get_class_label_dict():
    """
    get_class_label_dict()
    Description:
    Wrapper around returning a dictionary containing the signal modulation class labels

    Inputs:
    None

    Outputs:
    class_label_dict:dict - dictionary containing the signal modulation class labels
    """
    class_label_dict = {
        0:'32PSK',
        1:'16APSK',
        2:'32QAM',
        3:'FM',
        4:'GMSK',
        5:'32APSK',
        6:'OQPSK',
        7:'8ASK',
        8:'BPSK',
        9:'8PSK',
        10:'AM-SSB-SC',
        11:'4ASK',
        12:'16PSK',
        13:'64APSK',
        14:'128QAM',
        15:'128APSK',
        16:'AM-DSB-SC',
        17:'AM-SSB-WC',
        18:'64QAM',
        19:'QPSK',
        20:'256QAM',
        21:'AM-DSB-WC',
        22:'OOK',
        23:'16QAM'
    }
    return class_label_dict

def get_class_labels_strs(class_labels):
    class_label_dict = get_class_label_dict()
    class_label_strs = [class_label_dict[x] for x in class_labels]
    return class_label_strs

def preprocess_to_normal_set(signals, labels_int, snrs):
    """
    preprocess_to_normal_set(signals, labels_int, snrs)
    """
    class_labels_keep = get_class_labels_normal()
    signals, labels_int, snrs = trim_dataset_by_index(signals, labels_int, snrs, class_labels_keep)    
    labels = to_categorical(labels_int)
    labels = labels[:,class_labels_keep]
    labels = np.reshape(labels, (len(labels),1,len(class_labels_keep)))
    return

def trim_dataset_by_index(signals, labels, snrs, class_labels_keep = [3,4,6,8,9,11,16,17,19,22,23]):
    """
    def trim_to_normal_set(signals, labels, snrs, list_class_numbers)
    Take in a list of indices to keep from the dataset
    By default, this function creates a mask on to keep the following modulation types:
    Index | Mod. Type
    3     | FM
    4     | GMSK
    6     | OQPSK
    8     | BPSK
    9     | 8PSK
    11    | 4ASK
    16    | AM-DSB-SC
    17    | AM-SSB-SC
    19    | QPSK
    22    | OOK
    23    | 16QAM

    Inputs:
    signals:nparray - array of signal IQ data from deepsig io dataset
    labels:nparray - array of labels where each int corresponds to different modulation type (ref data_management.py)
    snrs:nparray - array of signal-to-noise ratios for each signal (pre-trimmed to >=10)
    """
    # default actions
    # class_labels_keep = [3,4,6,8,9,11,16,17,19,22,23]
    mask = np.zeros(len(labels),dtype="bool")
    for idx in class_labels_keep:
        masked_array = np.ma.masked_where(labels == idx, labels)
        mask |= masked_array.mask

    signals = signals[mask]
    labels = labels[mask]
    snrs = snrs[mask]
    return signals, labels, snrs

def preprocess_labels(labels_int, class_labels_keep):
    """
    preprocess_labels(labels_int, class_labels_keep)

    Description:
    This function converts the integer-based labels to one-hot encoded labels.
    Then, the one-hot encoded array is sliced to return the columns specified in class_labels_keep

    Inputs:
    labels_int - integer labels indicating the modulation type of a particular set of signal IQ data
    class_labels_keep - list of class labels, specified as integers, to keep.

    Outputs
    labels_oh - one-hot encoded version of labels_int
    """
    labels_oh = to_categorical(labels_int)
    labels_oh = labels_oh[:,class_labels_keep]
    return labels_oh

def get_class_labels_normal():
    """
    Returns indices corresponding to the following modulation types

    Index | Mod. Type
    3     | FM
    4     | GMSK
    6     | OQPSK
    8     | BPSK
    9     | 8PSK
    11    | 4ASK
    16    | AM-DSB-SC
    17    | AM-SSB-SC
    19    | QPSK
    22    | OOK
    23    | 16QAM
    """
    return [3,4,6,8,9,11,16,17,19,22,23]

# If running this file directly, just debug the process
if __name__ == '__main__':
    data_storage_dir = os.path.join(os.getcwd(),'data','project')
    # save_train_test_subset(data_storage_dir)
    pt1_storage_dir = os.path.join(data_storage_dir,"snr10_keep50_test10_seed1","pt1_data")
    pt2_storage_dir = os.path.join(data_storage_dir,"snr10_keep50_test10_seed1","pt2_data")
    pt3_storage_dir = os.path.join(data_storage_dir,"snr10_keep50_test10_seed1","pt3_data")
    signals_train, labels_int_train, snrs_train, signals_test, labels_int_test, snrs_test = load_train_test_subset(pt1_storage_dir)
    signals_train, labels_int_train, snrs_train, signals_test, labels_int_test, snrs_test = load_train_test_subset(pt2_storage_dir)
    signals_train, labels_int_train, snrs_train, signals_test, labels_int_test, snrs_test = load_train_test_subset(pt3_storage_dir)
    # signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = save_train_test_subset(data_storage_dir)
    # signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)
    pass
    