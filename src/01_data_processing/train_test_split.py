import os
import shutil
import sys
import random
import numpy as np

sys.path += ['src/01_data_processing']

from spectrogram import *

def make_train_test_split(base_dir,
                          db, machine_type, machine_id,
                          random_seed=None):
    normal_files = get_normal_mel_files(base_dir,
                                        db, machine_type, machine_id)
    abnormal_files = get_abnormal_mel_files(base_dir,
                                            db, machine_type, machine_id)
    if not random_seed is None:
        random.seed(random_seed)
    test_files = random.sample(normal_files, len(abnormal_files)) + abnormal_files
    train_files = [f for f in normal_files if f not in test_files]
    test_labels = [0] * len(abnormal_files) + [1] * len(abnormal_files)
    train_labels = [0] * len(train_files)

    return train_files, train_labels, test_files, test_labels


def subsample_from_mel(mel, dim, step):
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

    return batch


def generate_train_data(train_files, scaler, dim, step):
    for num, mel_file in enumerate(train_files):
        mel = np.load(mel_file)
        mel = apply_scaler_to_mel(scaler, mel)
        batch = subsample_from_mel(mel, dim, step)

        if num == 0:
            train_data = batch
        else:
            train_data = np.concatenate((train_data, batch))

    return train_data