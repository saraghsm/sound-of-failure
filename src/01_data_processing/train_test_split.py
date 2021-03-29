import os
import shutil
import sys
import random
import tqdm
import numpy as np

sys.path += ['src/01_data_processing']
from spectrogram import *


def make_train_test_split(base_dir,
                          db, machine_type, machine_id,
                          random_seed=None):
    """
    Generate train test split of mel spectrogram files.
    :param base_dir (str): path to directory where the "data" directory with mel spectrogram files is located
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.
    :param random_seed (int): seed for random split of normal mel spectrograms

    :return: mel file lists and labels for train and test set, respectively
    """
    normal_files = get_normal_mel_files(base_dir, db, machine_type, machine_id)
    abnormal_files = get_abnormal_mel_files(base_dir, db, machine_type, machine_id)
    if len(normal_files) == 0:
        return [], [], [], []
    elif len(abnormal_files) == 0:
        return [], [], [], []

    if not random_seed is None:
        random.seed(random_seed)
    test_files = random.sample(normal_files, len(abnormal_files)) + abnormal_files
    train_files = [f for f in normal_files if f not in test_files]
    test_labels = [0] * len(abnormal_files) + [1] * len(abnormal_files)
    train_labels = [0] * len(train_files)

    return train_files, train_labels, test_files, test_labels


def subsample_from_mel(mel, dim, step, as_images=True):
    """
    Generates a batch of small mel spectrograms from one (large) mel spectrogram.
    Subsamples are generated using a sliding time window.
    :param dim (int): dimension of time slices
    :param step (int): step of sliding window

    :return: feature vector (number of samples, dim, n_mels, 1)
    """
    mel = mel.T
    length = mel.shape[0]
    start_indices = np.arange(length - dim + step, step=step)

    for num, idx in enumerate(start_indices):
        start = min(length - dim, idx)
        one_slice = mel[start: start + dim, :]
        one_slice = one_slice.reshape((1, one_slice.shape[0], one_slice.shape[1]))

        if num == 0:
            batch = one_slice
        else:
            batch = np.concatenate((batch, one_slice))

    batch = batch.reshape((batch.shape[0],
                           batch.shape[1],
                           batch.shape[2],
                           1))
    if not as_images:
        batch = batch[:,:,:,0]

    return batch


def generate_train_data(train_files, scaler, dim, step, as_images=True):
    """
    Generates one large feature vector from a list of mel files.
    Feature batches are created by loading, scaling and subsampling mel spectrograms from the file list.
    :param train_files (list): list of mel spectrogram file paths (.npy)
    :param dim (int): dimension of time slices
    :param step (int): step of sliding window

    :return: feature vector (number of samples, dim, n_mels, 1)
    """
    for num, mel_file in tqdm.tqdm(enumerate(train_files), total=len(train_files)):
        mel = np.load(mel_file)
        mel = apply_scaler_to_mel(scaler, mel)
        batch = subsample_from_mel(mel, dim, step, as_images)

        if num == 0:
            train_data = batch
        else:
            train_data = np.concatenate((train_data, batch))

    if len(train_files) == 0:
        print('Cannot generate train data from empty file list.')
        return None

    return train_data