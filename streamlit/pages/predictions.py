##########################################################
# Import default Python libraries
##########################################################

import streamlit as st
import sys
import os
import glob
import numpy as np

##########################################################
# Import custom-made modules
##########################################################

sys.path += ['src/00_utils', 'src/01_data_processing', 
             'src/02_modelling', 'src/03_modell_evaluation',
             'src/04_visualization']
import naming
import spectrogram as spec
import train_model_autoencoder as train
import eval_model_autoencoder as eval
import visualize_autoencoder as vizae
import plotly_visualization as plotly_viz


##########################################################
# Tensorflow setup
##########################################################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

config = tf.ConfigProto(
    device_count={'CPU': 2},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

##########################################################
# Load config files
##########################################################

run_id = 'VAE_6dB_valve_id_00_final'
base_conf = train.read_config('conf/conf_base.ini')
conf_path = naming.get_conf_path(run_id)
param_conf = train.read_config(conf_path)

##########################################################
# Read data from config
##########################################################
# Data
BASE_DIR = "./" #base_conf['directories']['base_dir']
DB = param_conf.get('data', 'noise')
MACHINE_TYPE = param_conf.get('data', 'machine')
MACHINE_ID = param_conf.get('data', 'machine_id')
# Mel spectrograms
N_MELS = param_conf.getint('melspec', 'n_mels')
N_FFT = param_conf.getint('melspec', 'n_fft')
HOP_LENGTH = param_conf.getint('melspec', 'hop_length')
POWER = param_conf.getfloat('melspec', 'power')
WINDOW = param_conf.get('melspec', 'window')
# Subsampling
DIM = param_conf.getint('melspec', 'dim')
STEP = param_conf.getint('melspec', 'step')
# THRESHOLDING PARAMETERS
THRES_RANGE = np.arange(0.0,0.1,0.005)
REF_RANGE = 0.04

##########################################################
# Load trained model and fitted scaler
##########################################################
# Model
model_path = naming.get_model_path(run_id)
saved_model = train.load_saved_model(model_path)

# Scaler
path_saved_scaler = naming.get_scaler_path(run_id)
saved_scaler = spec.load_saved_scaler(path_saved_scaler)


##########################################################
# Load Reconstruction Loss of all training data
##########################################################

file_name = run_id + '_reco_loss_train.npy'
file_path = os.path.join(BASE_DIR, 'models', file_name)
reco_loss_train = np.load(file_path)

##########################################################
# Build page body and header
##########################################################
    
def build_header(header):
    header.title("Diagnose Machine Status")
    
def build_body(body):
    body.header("Use Reconstruction Error to Detect Machine Failure")
    body.write("A large reconstruction error indicates potential malfunction. The error varies over time. To "
               "diagnose the status of a machine, the error is averaged and compared to a threshold."
               "")
#    " and reconstructs the original Mel spectrogram from a latent space."
#    " Since our Autoencoder is trained on only normal sounds, it can"
#    " reconstruct well a sound from an unknwon/previously unseen normal machine state"
#    " with a small reconstruction error. If on the other hand an abnormal machine sound"
#    " is given as input the reconstruction error will be higher. We define a threshold on the reconstruction error, derived"
#    " from normal machine sounds. If the average reconstruction error is higher than the threshold, "
#    " we tag the data as anomaly indicating a machine malfunction.")
    
    # Load data
    #body.subheader("1. Data Loading")
    file = file_selector(body, BASE_DIR) 
    
    # Create Mel spectrogram under the hood
    mel_file = make_mels(wav_file=file,
                            n_mels=N_MELS, 
                            n_fft=N_FFT,
                            hop_length=HOP_LENGTH, 
                            base=BASE_DIR) 
                            
    # Predict Reconstruction loss
#    body.subheader("Diagnose Machine Status")
    show_results = st.checkbox('Show Diagnosis')
    if show_results:
        with session.as_default():
            with session.graph.as_default():
        
                fig = plotly_viz.make_eval_visualisation(mel_file, 
                                                        saved_model, 
                                                        saved_scaler,
                                                        reco_loss_train, 
                                                        DIM, 
                                                        STEP, 
                                                        THRES_RANGE, 
                                                        REF_RANGE, 
                                                        as_images=True)
                body.plotly_chart(fig)
    
    
    
def file_upload(container, type):
    file = container.file_uploader("Upload {} machine sound".format(type))
    return file
    
    
def file_selector(container, base, type='normal'):
    normal_wav_path = os.path.join(BASE_DIR, 'streamlit/data', '*.wav')
    normal_wav_files = sorted(glob.glob(os.path.join(normal_wav_path)))
    return container.selectbox('select', normal_wav_files, index=1)
    
    
def make_mels(wav_file, n_mels, n_fft, hop_length, base):
    mel = spec.mel_from_wav(wav_file,
                            n_mels=n_mels,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            power=2.0,
                            window='hann')
                            
    mel_save_path = os.path.join(base, 'streamlit', 'models', 'streamlit_mel.npy')
    np.save(mel_save_path, mel)
    return sorted(glob.glob(os.path.join(mel_save_path)))[0]