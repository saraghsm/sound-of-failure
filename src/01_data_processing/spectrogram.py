import sys
import os
import shutil
import librosa
import glob
import tqdm
import numpy as np
from librosa import display as ld
from IPython import display as ipd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals.joblib import load, dump

sys.path += ['src/00_utils']
from naming import *

def get_normal_wav_files(raw_data_dir, db, machine_type, machine_id):
    """
    Returns list of normal sound wav files for given noise level, machine type and id
    :param raw_data_dir (str): path to directory containing MIMII dataset
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: sorted list of wav files for normal sound
    """
    normal_dir = os.path.join(raw_data_dir, db, machine_type, machine_id, 'normal')
    if not os.path.exists(normal_dir):
        print(f'Directory {normal_dir} does not exist. Please make sure that raw data directory, ' + \
              'dB and machine type are correct and that machine ID exists.')
        return []

    return sorted(glob.glob(os.path.join(normal_dir, '*.wav')))


def get_abnormal_wav_files(raw_data_dir, db, machine_type, machine_id):
    """
    Returns list of abnormal sound wav files for given noise level, machine type and id
    :param raw_data_dir (str): path to directory containing MIMII dataset (wav files)
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: sorted list of wav files for abnormal sound
    """
    abnormal_dir = os.path.join(raw_data_dir, db, machine_type, machine_id, 'abnormal')
    if not os.path.exists(abnormal_dir):
        print(f'Directory {abnormal_dir} does not exist. Please make sure that raw data directory, ' + \
              'dB and machine type are correct and that machine ID exists.')
        return []

    return sorted(glob.glob(os.path.join(abnormal_dir, '*.wav')))


def make_mel_dirs(base_dir, db, machine_type, machine_id):
    """
    Generates a "data" directory with a "mel_spectrograms" subdirectory in the base directory.
    In "data/mel_spectrograms" a subdirectories like in the MIMII dataset are created, e.g.
    - "data/mel_spectrograms/6dB/valve/id_00/normal"
    - "data/mel_spectrograms/6dB/valve/id_00/abnormal"
    If directories exist they are not overwritten.

    :param base_dir (str): path to directory where a "data" directory shall be created
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: paths to directories for normal and abnormal mel spectrograms
    """
    data_dir = base_dir
    for dir in ['data', 'mel_spectrograms', db, machine_type, machine_id]:
        data_dir = os.path.join(data_dir, dir)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

    for dir in ['normal', 'abnormal']:
        mel_dir = os.path.join(data_dir, dir)
        if not os.path.exists(mel_dir):
            os.mkdir(mel_dir)
            print(f'Directory created: {mel_dir}')
        else:
            print(f'Directory already exists: {mel_dir}')


def get_mel_dirs(base_dir, db, machine_type, machine_id):
    """
    Retrieves the directories where normal and abnormal mel files are stored
    :param base_dir (str): path to directory where the "data" directory was or shall be created
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: paths to directories for normal and abnormal mel spectrograms
    """
    data_dir = base_dir
    for dir in ['data', 'mel_spectrograms', db, machine_type, machine_id]:
        data_dir = os.path.join(data_dir, dir)

    normal_dir = os.path.join(data_dir, 'normal')
    abnormal_dir = os.path.join(data_dir, 'abnormal')

    if not os.path.exists(normal_dir):
        print(f'Directory {normal_dir} does not exist.\n' + \
              'Please run make_mel_dirs(base_dir, db, machine_type, machine_id) to create it.')
        normal_dir = None

    if not os.path.exists(abnormal_dir):
        print(f'Directory {abnormal_dir} does not exist.\n' + \
              'Please run make_mel_dirs(base_dir, db, machine_type, machine_id) to create it.')
        abnormal_dir = None

    return normal_dir, abnormal_dir


def make_mels(raw_data_dir, base_dir,
              db, machine_type, machine_id,
              n_mels, n_fft, hop_length, power, window,
              overwrite=False):
    """
    Calculates mel spectrograms for all normal and abnormal sounds for a given specification
    of background noise, machine type and id. For each [core_file_name].wav file on spectrogram
    is generated and stored as [core_file_name].npy in the data directory that was created in
    the base directory.

    :param raw_data_dir (str): path to directory containing MIMII dataset (wav files)
    :param base_dir (str): path to directory where the "data" directory was created
    :param n_fft (int): no.of samples in each frame
    :param hop_length (int): hop samples
    :param n_mels (int): no. of mel-bands
    :param power (int): 1 for energy, 2 for power
    :param window (str): 'STFT' window, e.g. 'Hann'
    :param overwrite (bool): flag to control if existing files are overwritten
    """
    # Retrieve directories for storing mel spectrograms
    normal_dir, abnormal_dir = get_mel_dirs(base_dir, db, machine_type, machine_id)
    if (normal_dir is None) or (abnormal_dir is None):
        return
    save_dirs = [normal_dir, abnormal_dir]

    # Retrieve lists of wav files from which mel spectrograms will be generated
    wav_lists = [get_normal_wav_files(raw_data_dir, db, machine_type, machine_id),
                 get_abnormal_wav_files(raw_data_dir, db, machine_type, machine_id)]

    if (len(wav_lists[0]) == 0) or (len(wav_lists[1]) == 0):
        print('Did not find normal and abnormal wav files.')
        return

    # Loop for normal and abnormal sounds
    for save_dir, wav_list in zip(save_dirs, wav_lists):

        if save_dir == normal_dir:
            print(f'Generate normal spectrograms and save to {save_dir}.')
        else:
            print(f'Generate abnormal spectrograms and save to {save_dir}.')

        num_existing = 0
        num_created = 0
        for wav_file in tqdm.tqdm(wav_list):
            # Check if spectrogram file already exists
            mel_file = wav_file.split('/')[-1].replace('.wav', '.npy')
            mel_path = os.path.join(save_dir, mel_file)
            if os.path.exists(mel_path):
                if overwrite:
                    os.remove(mel_path)
                else:
                    # Check compatibility for 1 in 50 existing spectrograms
                    if num_existing % 50 == 0:
                        mel_loaded = np.load(mel_path)
                        mel_created = mel_from_wav(wav_file, n_mels, n_fft, hop_length, power, window)
                        assert (mel_loaded == mel_created).all(), \
                            'Existing spectrograms not compatible with n_mels, n_fft, hop_length, power, window. ' \
                            'Rerun with overwrite=True.'
                    num_existing += 1
                    continue

            # Generate spectrogram and save
            mel = mel_from_wav(wav_file, n_mels, n_fft, hop_length, power, window)
            np.save(mel_path, mel)
            num_created += 1

        print(f'Created {num_created} new spectrograms, kept {num_existing} existing spectrograms.')

def mel_from_wav(wav_file, n_mels, n_fft, hop_length, power, window):
    """
    Takes a wav file and converts it to a mel power spectrogram.
    :param wav_file (str): path to wav file that is converted
    :param n_fft (int): no.of samples in each frame
    :param hop_length (int): hop samples
    :param n_mels (int): no. of mel-bands
    :param power (int): 1 for energy, 2 for power
    :param window (str): 'STFT' window, e.g. 'Hann'

    :return: mel power spectrogram (numpy.ndarray)
    """
    y, sr = librosa.load(wav_file, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         n_mels=n_mels,
                                         power=power,
                                         window=window)
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel


def get_normal_mel_files(base_dir, db, machine_type, machine_id):
    """
    Returns list of normal sound spectrogram files (.npy) for given noise level, machine type and id
    :param raw_data_dir (str): path to directory containing MIMII dataset
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: sorted list of spectrogram files (.npy) for normal sound
    """
    normal_dir = os.path.join(base_dir, 'data', 'mel_spectrograms', db, machine_type, machine_id, 'normal')
    normal_mel_files = sorted(glob.glob(os.path.join(normal_dir, '*.npy')))
    if len(normal_mel_files) == 0:
        print(f'No mel spectrograms found in {normal_dir}.')
    return normal_mel_files


def get_abnormal_mel_files(base_dir, db, machine_type, machine_id):
    """
    Returns list of abnormal sound spectrogram files (.npy) for given noise level, machine type and id
    :param raw_data_dir (str): path to directory containing MIMII dataset
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: sorted list of spectrogram files (.npy) for abnormal sound
    """
    abnormal_dir = os.path.join(base_dir, 'data', 'mel_spectrograms', db, machine_type, machine_id, 'abnormal')
    abnormal_mel_files = sorted(glob.glob(os.path.join(abnormal_dir, '*.npy')))
    if len(abnormal_mel_files) == 0:
        print(f'No mel spectrograms found in {abnormal_dir}.')
    return abnormal_mel_files


def create_scaler(scaler_type):
    """
    Creates a normalizer/scaler
    :param scaler_type (str): type of scaler to be created

    :return: scaler (sklearn.preprocessing obj)
    """
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        print('Invalid scaler_type. Choose StandardScaler or MinMaxScaler')
    return scaler


def fit_scaler_to_mel_files(scaler, file_list):
    """
    Function for fitting a normalizer/scaler (sklearn.preprocessing) to multiple spectrogram files.
    :param scaler (sklearn.preprocessing obj): scaler that is fitted
    :param file_list (str): list of spectrogram files (.npy)
    """
    for mel_file in tqdm.tqdm(file_list):
        mel = np.load(mel_file)
        flat_mel = mel.flatten().reshape(-1, 1)
        scaler.partial_fit(flat_mel)
    if len(file_list) == 0:
        print('Cannot fit scaler to empty file list.')
        return


def fit_and_save_scaler(scaler_type, file_list, scaler_path, overwrite=False):
    """
    Function for fitting a normalizer/scaler (sklearn.preprocessing) to multiple spectrogram files.
    The result is stored locally under a default name according to name_string.
    Before fitting it is checked if a stored scaler is found.
    By default the stored scaler is loaded and returned, instead of fitting again .
    :param scaler_type (str): type of scaler to be created
    :param file_list (str): list of spectrogram files (.npy)
    :param scaler_path (str): scaler path used for storing/loading
    :overwrite (bool): flag to control if existing scaler is overwritten instead of loaded

    :return: fitted scaler
    """
    if os.path.exists(scaler_path):
        if overwrite:
            print(f'Overwriting existing scaler {scaler_path}.')
        else:
            print(f'Loading existing scaler {scaler_path}.')
            return load(scaler_path)

    scaler = create_scaler(scaler_type)
    fit_scaler_to_mel_files(scaler, file_list)
    print(f'Saving scaler to {scaler_path}.')
    dump(scaler, scaler_path)

    return scaler


def apply_scaler_to_mel(scaler, mel, inverse=False):
    """
    Function for applying a fitted normalizer/scaler (sklearn.preprocessing) to a single mel spectrogram.
    :param scaler (sklearn.preprocessing obj): scaler that is applied
    :param mel (numpy.ndarray): mel spectrogram

    :return: scaled mel spectrogram (numpy.ndarray)
    """
    shape_ = mel.shape
    flat_mel = mel.flatten().reshape(-1, 1)
    if inverse:
        scaled_flat_mel = scaler.inverse_transform(flat_mel)
    else:
        scaled_flat_mel = scaler.transform(flat_mel)
    scaled_mel = scaled_flat_mel.reshape(shape_)
    return scaled_mel


def load_saved_scaler(scaler_path):
    """
    Load the saved scaler from scaler_path.
    This is useful to load an already fitted scaler
    if it exists.
    """
    if os.path.exists(scaler_path):
      loaded_scaler = load(scaler_path)
    else:
      print("Path {} to scaler does not exist. Exiting...".format(scaler_path))
    return loaded_scaler
