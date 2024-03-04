# EENG 645A - Final Project
# build_tf_records.py
# Build a file to handle the TF Records aspect of project
# Uses instructor-provided code for Lab5, "lab5transfer.py"

import datetime
import itertools
import os
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics
import tensorflow as tf
from tensorflow.keras.applications import resnet
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.metrics import BinaryAccuracy, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, Precision, Recall
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm
from tqdm.keras import TqdmCallback

import time


# # The following functions can be used to convert a value to a type compatible
# # with tf.train.Example.

# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# # this function was added since the _int64_feature takes a single int and not a list of ints
# def _int64_list_feature(value: typing.List[int]):
#     """Returns an int64_list from a list of bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# # Function for reading signal IQs from disk and writing them along with the class-labels to a TFRecord file.
# def convert(x_data: np.ndarray,
#             y_data: np.ndarray,
#             out_path: Path,
#             records_per_file: int = 4500,
#             ) -> typing.List[Path]:
#     """
#     Function for reading IQs from disk and writing them along with the class-labels to a TFRecord file.

#     :param x_data: the input, feature data to write to disk
#     :param y_data: the output, label, truth data to write to disk
#     :param out_path: File-path for the TFRecords output file.
#     :param records_per_file: the number of records to use for each file
#     :return: the list of tfrecord files created
#     """

#     if not os.path.exists(out_path):
#         os.makedirs(out_path)

#     # Open a TFRecordWriter for the output-file.
#     record_files = []
#     n_samples = x_data.shape[0]
#     # Iterate over all the image-paths and class-labels.
#     n_tfrecord_files = int(np.ceil(n_samples / records_per_file))
#     for idx in tqdm(range(n_tfrecord_files),
#                     desc="Convert Batch",
#                     total=n_tfrecord_files,
#                     position=0):
#         record_file = out_path / f'train{idx}.tfrecord'
#         record_files.append(record_file)
#         slicer = slice(idx * records_per_file, (idx + 1) * records_per_file)
#         with tf.io.TFRecordWriter(str(record_file)) as writer:
#             for x_sample, y_sample in tqdm(zip(x_data[slicer], y_data[slicer]),
#                                            desc="Convert Signal IQ in batch",
#                                            total=records_per_file,
#                                            position=1,
#                                            leave=False):
#                 # Convert the ndarray of the image to raw bytes. note this is bytes encodes as uint8 types
#                 img_bytes = x_sample.tostring()
#                 # Create a dict with the data we want to save in the
#                 # TFRecords file. You can add more relevant data here.
#                 data = {
#                     'signal_iq': _bytes_feature(img_bytes),
#                     'label': _int64_list_feature(y_sample)
#                 }
#                 # Wrap the data as TensorFlow Features.
#                 feature = tf.train.Features(feature=data)
#                 # Wrap again as a TensorFlow Example.
#                 example = tf.train.Example(features=feature)
#                 # Serialize the data.
#                 serialized = example.SerializeToString()
#                 # Write the serialized data to the TFRecords file.
#                 writer.write(serialized)
#     return record_files

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

    # SNR range: -20:2:
    
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
# def get_dataset(filenames: typing.List[Path], data_shape: tuple) -> tf.data.Dataset:
#     """
#     This function takes the filenames of tfrecords to process into a dataset object
#     The _parse_function takes a serialized sample pulled from the tfrecord file and
#     parses it into a sample with x (input) and y (output) data, thus a full sample for training

#     This function will not do any scaling, batching, shuffling, or repeating of the dataset

#     :param filenames: the file names of each tf record to process
#     :param data_shape: the size of the data in width, height, channels
#     :return: the dataset object made from the tfrecord files and parsed to return samples
#     """

#     def _parse_function(serialized):
#         """
#         This function parses a serialized object into tensor objects to use for training
#         NOTE: you must use tensorflow functions in this section
#         using non-tensorflow function will not get the results you expect and/or hinder performance
#         see how each function starts with `tf.` meaning the function is form the tensorflow library
#         """
#         features = {
#             'signal_iq': tf.io.FixedLenFeature([], tf.string),
#             'label': tf.io.FixedLenFeature([], tf.int64)
#         }
#         # Parse the serialized data so we get a dict with our data.
#         parsed_example = tf.io.parse_single_example(serialized=serialized,
#                                                     features=features)
#         # convert the data's shape to a tensorflow object
#         data_shape = tf.stack(img_shape)

#         # get the raw feature bytes
#         data_raw = parsed_example['signal_iq']
#         # Decode the raw bytes so it becomes a tensor with type.
#         data_inside = tf.io.decode_raw(data_raw, tf.uint8)
#         # cast to float32 for GPU operations
#         data_inside = tf.cast(data_inside, tf.float32)
#         # reshape to correct image shape
#         data_inside = tf.reshape(data_inside, data_shape)

#         # get the label and convert it to a float32
#         label = tf.cast(parsed_example['label'], tf.float32)

#         # return a single tuple of the (features, label)
#         return data_inside, label

#     # the tf functions takes string names not path objects, so we have to convert that here
#     filenames_str = [str(filename) for filename in filenames]
#     # make a dataset from slices of our file names
#     files_dataset = tf.data.Dataset.from_tensor_slices(filenames_str)

#     # make an interleaved reader for the TFRecordDataset files
#     # this will give us a stream of the serialized data interleaving from each file
#     dataset = files_dataset.interleave(map_func=lambda x: tf.data.TFRecordDataset(x),
#                                        # 12 was picked for the cycle length because there were 12 total files,
#                                        # and I wanted to cycle through all of them
#                                        cycle_length=12,  # how many files to cycle through at once
#                                        block_length=1,  # how many samples from each file to get
#                                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
#                                        deterministic=False)

#     # Parse the serialized data in the TFRecords files.
#     # This returns TensorFlow tensors for the image and labels.
#     dataset = dataset.map(map_func=_parse_function,
#                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     return dataset

def main():
    data_storage_dir = os.path.join(os.getcwd(),'data','project')
    # signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = save_train_test_subset(data_storage_dir)
    signals_train_full, labels_int_train_full, snrs_train_full, signals_test, labels_int_test, snrs_test = load_train_test_subset(data_storage_dir)
    return



if __name__ == '__main__':
    main()