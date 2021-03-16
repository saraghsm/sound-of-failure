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
    normal_dir = os.path.join(raw_data_dir,
                              db, machine_type, machine_id, 'normal')
    return sorted(glob.glob(os.path.join(normal_dir, '*.wav')))


def get_abnormal_wav_files(raw_data_dir, db, machine_type, machine_id):
    normal_dir = os.path.join(raw_data_dir,
                              db, machine_type, machine_id, 'abnormal')
    return sorted(glob.glob(os.path.join(normal_dir, '*.wav')))


def make_mel_dirs(base_dir, db, machine_type, machine_id):
    data_dir = base_dir
    for dir in ['data', 'mel_spectrograms', db, machine_type, machine_id]:
        data_dir = os.path.join(data_dir, dir)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

    final_dirs = []
    for dir in ['normal', 'abnormal']:
        final_dirs.append(os.path.join(data_dir, dir))
        if not os.path.exists(os.path.join(data_dir, dir)):
            os.mkdir(os.path.join(data_dir, dir))

    return tuple(final_dirs)


def make_mels(raw_data_dir, base_dir,
              db, machine_type, machine_id,
              n_mels, n_fft, hop_length, power, window):
    normal_dir, abnormal_dir = make_mel_dirs(base_dir,
                                             db, machine_type, machine_id)
    save_dirs = [normal_dir, abnormal_dir]
    wav_lists = [get_normal_wav_files(raw_data_dir,
                                      db, machine_type, machine_id),
                 get_abnormal_wav_files(raw_data_dir,
                                        db, machine_type, machine_id)]

    for save_dir, wav_list in zip(save_dirs, wav_lists):

        if save_dir == normal_dir:
            print(f'Generate normal spectrograms and save to {save_dir}.')
        else:
            print(f'Generate abnormal spectrograms and save to {save_dir}.')

        for wav_file in wav_list:
            y, sr = librosa.load(wav_file, sr=None, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels,
                                                 power=power,
                                                 window=window)
            mel = librosa.power_to_db(mel, ref=np.max)
            mel_file = wav_file.split('/')[-1].replace('.wav', '.npy')
            mel_path = os.path.join(save_dir, mel_file)
            np.save(mel_path, mel)


def get_normal_mel_files(base_dir, db, machine_type, machine_id):
    normal_dir = os.path.join(base_dir, 'data', 'mel_spectrograms',
                              db, machine_type, machine_id, 'normal')
    return sorted(glob.glob(os.path.join(normal_dir, '*.npy')))


def get_abnormal_mel_files(base_dir, db, machine_type, machine_id):
    abnormal_dir = os.path.join(base_dir, 'data', 'mel_spectrograms',
                                db, machine_type, machine_id, 'abnormal')
    return sorted(glob.glob(os.path.join(abnormal_dir, '*.npy')))


def create_scaler(scaler_type):
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        print('Invalid scaler_type. Choose StandardScaler or MinMaxScaler')
    return scaler


def fit_scaler_to_mel_files(scaler, file_list):
    for mel_file in file_list:
        mel = np.load(mel_file)
        flat_mel = mel.flatten().reshape(-1, 1)
        scaler.partial_fit(flat_mel)


def apply_scaler_to_mel(scaler, mel):
    shape_ = mel.shape
    flat_mel = mel.flatten().reshape(-1, 1)
    flat_mel = scaler.transform(flat_mel)
    return flat_mel.reshape(shape_)