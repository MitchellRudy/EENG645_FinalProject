# EENG 645A - Final Project
# data_management.py

import os
import numpy as np
from sklearn.model_selection import train_test_split

import time

def save_train_test_subset(base_dir: str, snr_thresh = 10, seed = 1, tts_percentage = 0.90):
    """
    save_train_test_subset(base_dir: Path, snr_thresh = 10, seed = 1, tts_percentage = 0.90)

    Description:
    Load in the full deepsig_io_radioml_2018_01a_dataset dataset, then trim it down based on the snr_thresh value.
    Split remaining dataset into a training and test dataset based on the tts_percentage.

    Inputs:
    base_dir: str - Path to directory where project's numpy arrays shall be saved
    snr_thresh=10 - Lower bound on snr range kept and saved into new arrays for project
    seed=1 - seed to control train/test splitting randomness
    tts_percentage=0.90 - percentage of dataset to keep in the training subset

    Outputs:
    signals_train_full: nparray - Numpy array containing IQ arrays for train+validation signals
    labels_int_train_full: nparray - Numpy array containing classification labels for train+validation signals
    snrs_train_full: nparray - Numpy array containing snrs for train+validation signals
    signals_test: nparray - Numpy array containing IQ arrays for test signals
    labels_int_test: nparray - Numpy array containing classification labels for test signals
    snrs_test: nparray - Numpy array containing snrs for test signals
    """
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
    
    # Make base directory for data
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # make folders for train and test
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # File names for train and test data
    signals_train_saveloc = os.path.join(train_dir, 'signals.npy')
    signals_test_saveloc = os.path.join(test_dir, 'signals.npy')
    labels_train_saveloc = os.path.join(train_dir, 'labels.npy')
    labels_test_saveloc = os.path.join(test_dir, 'labels.npy')
    snrs_train_saveloc = os.path.join(train_dir, 'snrs.npy')
    snrs_test_saveloc = os.path.join(test_dir, 'snrs.npy')
    
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

    signals = signals[snr_mask]
    snrs = snrs[snr_mask]
    labels_int = labels_int[snr_mask]

    # Perform train test split on indices
    # https://stackoverflow.com/questions/31467487/memory-efficient-way-to-split-large-numpy-array-into-train-and-test
    indices = np.arange(len(signals))
    indices_train_full, indices_test, labels_int_train_full, labels_int_test = train_test_split(indices, labels_int, random_state=seed, train_size=tts_percentage)
    
    signals_train_full = signals[indices_train_full]
    snrs_train_full = snrs[indices_train_full]

    signals_test = signals[indices_test]
    snrs_test = snrs[indices_test]

    print("Split dataset")

    # Save the dataset
    np.save(signals_train_saveloc, signals_train_full)
    np.save(labels_train_saveloc, labels_int_train_full)
    np.save(snrs_train_saveloc, snrs_train_full)
    np.save(signals_test_saveloc, signals_test)
    np.save(labels_test_saveloc, labels_int_test)
    np.save(snrs_test_saveloc, snrs_test)

    print(f"saved datasets to {base_dir}")

    return signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test

def load_train_test_subset(base_dir):
    """
    load_train_test_subset(base_dir):
    
    Description:
    Load the numpy arrays used for training and testing the models.
    To create a subset, use "save_train_test_subset"

    Inputs:
    base_dir: str - Path to directory where project's numpy arrays are stored

    Outputs:
    signals_train_full: nparray - Numpy array containing IQ arrays for train+validation signals
    labels_int_train_full: nparray - Numpy array containing classification labels for train+validation signals
    snrs_train_full: nparray - Numpy array containing snrs for train+validation signals
    signals_test: nparray - Numpy array containing IQ arrays for test signals
    labels_int_test: nparray - Numpy array containing classification labels for test signals
    snrs_test: nparray - Numpy array containing snrs for test signals
    """
    # Define the train and test dataset subdirectories
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    # check all expected  directories exist
    if not (os.path.exists(base_dir) or os.path.exists(train_dir) or os.path.exists(test_dir)):
        print(f"At least one expected directory do not exist. Returning None")
        return None, None, None, None, None, None
    # Load in the full training data
    signals_train_full = np.load(os.path.join(train_dir,'signals.npy'))
    labels_int_train_full = np.load(os.path.join(train_dir,'labels.npy'))
    snrs_train_full = np.load(os.path.join(train_dir,'snrs.npy'))
    # Load in the test data
    signals_test= np.load(os.path.join(test_dir,'signals.npy'))
    labels_int_test = np.load(os.path.join(test_dir,'labels.npy'))
    snrs_test = np.load(os.path.join(test_dir,'snrs.npy'))


    return signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test

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

def trim_dataset_by_index(signals, labels, snrs, class_labels_keep):
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

def get_class_labels_normal():
    return [3,4,6,8,9,11,16,17,19,22,23]

# If running this file directly, just debug the process
if __name__ == '__main__':
    data_storage_dir = os.path.join(os.getcwd(),'data','project')
    signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = save_train_test_subset(data_storage_dir)
    signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)
    