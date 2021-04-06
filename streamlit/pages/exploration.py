##########################################################
# Import default Python libraries
##########################################################
import streamlit as st
import librosa
from librosa import display as ld
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob

##########################################################
# Import custom-made modules
##########################################################
sys.path += ['src/00_utils', 'src/01_data_processing', 'src/02_modelling',
             'src/04_visualization', 'src/03_modell_evaluation']
import spectrogram as spec
import visualize_autoencoder as vizae
import train_model_autoencoder as train


##########################################################
# Read data from config
##########################################################
run_id = 'VAE_6dB_valve_id_00_final'
base_conf = train.read_config('conf/conf_base.ini')
BASE_DIR = "./" #base_conf['directories']['base_dir']

##########################################################
# Build page body and header
##########################################################

def upload_image(container, img):
    container.image(img, width=700)
    

def build_header(header):
    header.title("Comparison of Normal and Abnormal Sounds")
    


def build_body(body):
    body.header("Audio and Mel Spectrogram")
#    body.write("We convert the audio problem into "
#               " a computer vision problem, i.e."
#               " we 'see' the sound! We convert"
#               " the raw waveform into its time-frequency"
#               " representation known as Mel spectrograms."
#               " Then we extract features from Mel spectrograms"
#               " while training and evaluating using "
#               " Machine Learning models.")

    normal_col, abnormal_col = body.beta_columns(2)

    # Choose a normal sound
    normal_col.subheader("Normal Sound")
    file_normal = file_selector(normal_col, BASE_DIR, type='normal') 

    # Choose an abnormal sound
    abnormal_col.subheader("Abnormal Sound")
    file_abnormal = file_selector(abnormal_col, BASE_DIR, type='abnormal')

    # Play normal sound
    #normal_col.subheader("Play Sound")
    sound_normal = play_sound(normal_col, file_normal)

    # Play abnormal sound
    #abnormal_col.subheader("Play Sound")
    sound_abnormal = play_sound(abnormal_col, file_abnormal)

    # Mel spectrograms normal machine
    #normal_col.subheader("Mel Spectrogram")
    n_fft_norm = normal_col.selectbox('n_fft', [256, 512, 1024], index=2)
    n_mels_norm = normal_col.selectbox('n_mels', [16, 32, 64, 128], index=1)

    # Mel spectrograms abnormal machine
    #abnormal_col.subheader("Mel Spectrogram")
    n_fft_abnorm = abnormal_col.selectbox('n_fft', [256, 512, 1024], index=1)
    n_mels_abnorm = abnormal_col.selectbox('n_mels', [16, 32, 64, 128], index=2)

    # Submit button
    submit_mel_params_norm = normal_col.button("Create Mel Spectrogram")
    submit_mel_params_abnorm = abnormal_col.button("Create Mel Spectrogram ")
    if submit_mel_params_norm or submit_mel_params_abnorm:

        mel_norm = spec.mel_from_wav(file_normal,
                                     int(n_mels_norm),
                                     int(n_fft_norm),
                                     int(n_fft_norm * 0.5),
                                     power=2.0,
                                     window='hann')

        mel_abnorm = spec.mel_from_wav(file_abnormal,
                                     int(n_mels_abnorm),
                                     int(n_fft_abnorm),
                                     int(n_fft_abnorm * 0.5),
                                     power=2.0,
                                     window='hann')

        fig_norm = vizae.plot_spectrogram(mel_norm, 16000, ndarray=True)
        normal_col.pyplot(fig_norm)

        fig_abnorm = vizae.plot_spectrogram(mel_abnorm, 16000, ndarray=True)
        abnormal_col.pyplot(fig_abnorm)

def upload_image(container, img):
    container.image(img, width=300)
    
def file_upload(container, type):
    file = container.file_uploader("Upload {} machine sound".format(type))
    return file
    
    
def file_selector(container, base, type='normal'):
    normal_wav_path = os.path.join(base, 'streamlit/data', type+'_*.wav')
    normal_wav_files = sorted(glob.glob(os.path.join(normal_wav_path)))
    return container.selectbox('select', normal_wav_files, index=1)


def play_sound(container, file):
    sound = container.audio(file)
    return sound






