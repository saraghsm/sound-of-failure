##########################################################
# Import default Python libraries
##########################################################

import streamlit as st
import sys
import os
import glob
import matplotlib.pyplot as plt
from librosa import display as ld


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
# Import custom-made modules
##########################################################

sys.path += ['src/00_utils', 'src/01_data_processing', 'src/02_modelling',
            'src/03_modell_evaluation', 'src/04_visualization',]
import spectrogram as spec
import train_model_autoencoder as train
import eval_model_autoencoder as eval
import visualize_autoencoder as vizae
import naming

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
# Mel spectrograms
N_MELS = param_conf.getint('melspec', 'n_mels')
N_FFT = param_conf.getint('melspec', 'n_fft')
HOP_LENGTH = param_conf.getint('melspec', 'hop_length')
POWER = param_conf.getfloat('melspec', 'power')
WINDOW = param_conf.get('melspec', 'window')
# Subsampling
DIM = param_conf.getint('melspec', 'dim')
STEP = param_conf.getint('melspec', 'step')

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
# Build page body and header
##########################################################

def build_header(header):
    header.title("AI-Based Sound Analysis")

def build_body(body):
    body.header("Detecting abnormal sounds")
    body.write("We use a deep autoencoder that is trained to reconstruct sounds of normal functioning machines. "
               "Abnormal sounds are reconstructed worse. This indicates a potential machine failure.")

    #body.write("We have built an AI for detecting machine malfunctions from sound which works on the principles"
    #           " of unsupervised sound anomaly detection."
    #           " We use Autoencoders for this purpose. The sounds converted to Mel spectrograms"
    #           " is normalized and broken down into smaller chunks by subsampling."
    #           " The samples are then passed through the trained Autoencoder which then"
    #           " reconstructs the input spectrogram. If the reconstruction is good,"
    #           " the reconstruction error (the squared difference between the input and"
    #           " reconstructed spectrograms) will be small and large otherwise."
    #           " Small reconstruction error corresponds to mostly black pixels in the"
    #           " reconstruction error plot.")

    # Load data
    #body.subheader("1. Data Loading")
    file = file_selector(body, BASE_DIR) 
    
    # Button to do analysis
    do_analysis = body.button("Perform Analysis")
    
    # Preprocess data
    body.subheader("1. Input Data")
    
    ##########################################################
    # Create Mel spectrograms
    ##########################################################
    # Define Mel containers
    mel_container = body.beta_container()

    # Mel spectrogram generation
    #mel_container.subheader("1.1. Create Mel-Spectrograms")
    mel_container.write("Mel spectrogram")
    mel = spec.mel_from_wav(file,
                            n_mels=N_MELS,
                            n_fft=N_FFT,
                            hop_length=HOP_LENGTH,
                            power=2.0,
                            window='hann')
    if do_analysis:
        fig = vizae.plot_spectrogram(mel, 16000, ndarray=True)
        mel_container.pyplot(fig)
        
    ##########################################################
    # Normalize Mel spectrograms
    ##########################################################
    # Define Scale containers
    #scale_container = body.beta_container()

    # Scale Mel spectrogram
    #scale_container.subheader("1.2. Normalize Mel-Spectrograms")
    #scale_container.write("We normalize the Mel spectrograms with zero mean and unit standard deviation.")
    #scaled_mel = spec.apply_scaler_to_mel(saved_scaler, mel)
    
    #if do_analysis:
    #        fig2 = vizae.plot_spectrogram(scaled_mel, 16000, ndarray=True)
    #        scale_container.pyplot(fig2)
            
    ##########################################################
    # Subsamle Mel spectrograms
    ##########################################################
    # Find original and reconstrcuted subsamples of spectrogram
    with session.as_default():
        with session.graph.as_default():
            orig, decoded = eval.decode_spectrogram(saved_model,
                                                    saved_scaler,
                                                    DIM,
                                                    STEP,
                                                    mel,
                                                    ndarray=True)
                                                    
    # Define subsample containers
    subsample_container = body.beta_container()
    # Subsample Mel spectrogram
    #subsample_container.subheader("Subsampling")
    #subsample_mel = subsample_container.button("Subsample Mel")
    subsample_container.write("Subsampling (sliding 1 sec time window)")
    if do_analysis:
            fig3 = render_subsamples(orig, sr=16000)
            subsample_container.pyplot(fig3)
    
    ##########################################################
    # Decode Mel spectrograms
    ##########################################################
    # Define Prediction containers
    pred_container = body.beta_container()
    
    # Model predictions
    pred_container.subheader("2. Sound Reconstruction with Autoencoder")
    pred_container.write("The autoencoder produces a reconstruction of each subsample. "
                         "The reconstruction error is the squared difference between input and output. "
                         "Low values (black pixels) indicate good reconstruction.")

    #           " reconstructs the input spectrogram. If the reconstruction is good,"
    #           " the reconstruction error (the squared difference between the input and"
    #           " reconstructed spectrograms) will be small and large otherwise."
    #           " Small reconstruction error corresponds to mostly black pixels in the"
    #           " reconstruction error plot.")

    # Batch predictions
    #pred_container.subheader("3.1. Comparison of the original and reconstructed spectrograms")
    
    #decode_mel = pred_container.button("Decode Mel")
    if do_analysis:
                fig4 = vizae.render_predictions(orig, decoded, sr=16000)
                pred_container.pyplot(fig4)


##########################################################
# Function definitions
##########################################################
def file_upload(container, type):
    file = container.file_uploader("Upload {} machine sound".format(type))
    return file
    

def file_selector(container, base, type='normal'):
    normal_wav_path = os.path.join(BASE_DIR, 'streamlit/data', '*_00000000.wav')
    normal_wav_files = sorted(glob.glob(os.path.join(normal_wav_path)))
    return container.selectbox('select', normal_wav_files, index=1)
    
    
def render_subsamples(original_spec, sr):
    """
    Given the original spectrogram slices
    (the slices come by subsampling a bigger spectrogram),
    plot them.
    """
    fig, axs = plt.subplots(1, 10, figsize=(9, 3))
    

    for index in range(10):
        image_in = original_spec[index].reshape(original_spec[0].shape[0],
                                                original_spec[0].shape[1])

        orig = ld.specshow(image_in.T,
                           x_axis='time',
                           y_axis='mel',
                           sr=sr,
                           fmax=sr * 0.5,
                           ax=axs[index],
                           cmap='inferno')


        axs[index].set_title("Original", fontsize=8)
        axs[index].axis('off')
    return fig