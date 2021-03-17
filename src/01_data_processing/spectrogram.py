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


def get_normal_wav_files(raw_data_dir, db, machine_type, machine_id):
    """
    Returns list of normal sound wav files for given noise level, machine type and id
    :param raw_data_dir (str): path to directory containing MIMII dataset
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: sorted list of wav files for normal sound
    """
    normal_dir = os.path.join(raw_data_dir,
                              db, machine_type, machine_id, 'normal')
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
    normal_dir = os.path.join(raw_data_dir,
                              db, machine_type, machine_id, 'abnormal')
    return sorted(glob.glob(os.path.join(normal_dir, '*.wav')))


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
    dirs_exist = True
    data_dir = base_dir
    for dir in ['data', 'mel_spectrograms', db, machine_type, machine_id]:
        data_dir = os.path.join(data_dir, dir)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            dirs_exist = False

    final_dirs = []
    for dir in ['normal', 'abnormal']:
        final_dirs.append(os.path.join(data_dir, dir))
        if not os.path.exists(os.path.join(data_dir, dir)):
            os.mkdir(os.path.join(data_dir, dir))
            dirs_exist = False

    normal_dir = final_dirs[0]
    abnormal_dir = final_dirs[1]
    if dirs_exist:
        print(f'Directories already exist.\nNormal: {normal_dir}\nAbnormal: {abnormal_dir}')
    else:
        print(f'Directories created.\nNormal: {normal_dir}\nAbnormal: {abnormal_dir}')

    return normal_dir, abnormal_dir


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
    normal_dir, abnormal_dir = get_mel_dirs(base_dir, db, machine_type, machine_id)
    if not os.path.exists(normal_dir):
        print(f'Directory {normal_dir} does not exist.')
        print('Please run make_mel_dirs(base_dir, db, machine_type, machine_id) to create all required directories.')
    if not os.path.exists(abnormal_dir):
        print(f'Directory {abnormal_dir} does not exist.')
        print('Please run make_mel_dirs(base_dir, db, machine_type, machine_id) to create all required directories.')

    save_dirs = [normal_dir, abnormal_dir]
    wav_lists = [get_normal_wav_files(raw_data_dir, db, machine_type, machine_id),
                 get_abnormal_wav_files(raw_data_dir, db, machine_type, machine_id)]

    for save_dir, wav_list in zip(save_dirs, wav_lists):

        if save_dir == normal_dir:
            print(f'Generate normal spectrograms and save to {save_dir}.')
        else:
            print(f'Generate abnormal spectrograms and save to {save_dir}.')

        for wav_file in tqdm.tqdm(wav_list):
            mel_file = wav_file.split('/')[-1].replace('.wav', '.npy')
            mel_path = os.path.join(save_dir, mel_file)
            if os.path.exists(mel_path) and not overwrite:
                print(f'File already exists: {mel_path}')
                continue

            y, sr = librosa.load(wav_file, sr=None, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels,
                                                 power=power,
                                                 window=window)
            mel = librosa.power_to_db(mel, ref=np.max)
            np.save(mel_path, mel)


def get_normal_mel_files(base_dir, db, machine_type, machine_id):
    """
    Returns list of normal sound spectrogram files (.npy) for given noise level, machine type and id
    :param raw_data_dir (str): path to directory containing MIMII dataset
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: sorted list of spectrogram files (.npy) for normal sound
    """
    normal_dir = os.path.join(base_dir, 'data', 'mel_spectrograms',
                              db, machine_type, machine_id, 'normal')
    return sorted(glob.glob(os.path.join(normal_dir, '*.npy')))


def get_abnormal_mel_files(base_dir, db, machine_type, machine_id):
    """
    Returns list of abnormal sound spectrogram files (.npy) for given noise level, machine type and id
    :param raw_data_dir (str): path to directory containing MIMII dataset
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: sorted list of spectrogram files (.npy) for abnormal sound
    """
    abnormal_dir = os.path.join(base_dir, 'data', 'mel_spectrograms',
                                db, machine_type, machine_id, 'abnormal')
    return sorted(glob.glob(os.path.join(abnormal_dir, '*.npy')))


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


def apply_scaler_to_mel(scaler, mel):
    """
    Function for applying a fitted normalizer/scaler (sklearn.preprocessing) to a single mel spectrogram.
    :param scaler (sklearn.preprocessing obj): scaler that is applied
    :param mel (numpy.ndarray)

    :return: scaled mel spectrogram
    """
    shape_ = mel.shape
    flat_mel = mel.flatten().reshape(-1, 1)
    scaled_flat_mel = scaler.transform(flat_mel)
    scaled_mel = scaled_flat_mel.reshape(shape_)
    return scaled_mel
