##########################################################
# Import default Python libraries
##########################################################
from librosa import display as ld
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython import display as ipd
import sys
import os
import shutil
import librosa
import glob
import tqdm
import json
import ast
import random
import joblib
from configparser import ConfigParser

##########################################################
# Import tensorflow libraries
##########################################################
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
tf.compat.v1.disable_eager_execution()

##########################################################
# Import sklearn libraries
##########################################################
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import load, dump
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

##########################################################
# Import custom-made modules
##########################################################
sys.path += ['src/01_data_processing', 'src/02_modelling', 'TEST/']
import spectrogram as spec
import train_test_split as splt
from conv_autoencoder import ConvAutoencoder
from var_autoencoder import VarAutoencoder

# NOTE : Before running the training
# pip install git+https://github.com/AI-Guru/ngdlm.git
# IMPORTANT for the VAE.

##########################################################
# Reading the config file for parameters
##########################################################
def read_config(config_file):
    """
    Function for reading config files
    config_file (str): Path to the config
      Config files stay inside conf/.
    """
    config = ConfigParser()
    config.read(config_file)
    return config

##########################################################
# Load the Training model from scratch
##########################################################
def load_new_model(model_name,
               input_shape,
               num_nodes,
               num_kernel,
               num_strides,
               latent_dim):
    """
    model_name : AE or VAR
    input_shape : Shape of spectrogram slice
    num_nodes: Nodes in each convolutional layer
    num_kernel: Size of the kernel in each
                convolutional layer
    num_strides: No. of strides in each convolution
    """
    if model_name == 'AE':
        model = ConvAutoencoder(input_shape=input_shape,
                                num_nodes=num_nodes,
                                num_kernel=num_kernel,
                                num_strides=num_strides,
                                latent_dim=latent_dim)
        return model.model
    elif model_name == 'VAE':
        model = VarAutoencoder(input_shape=input_shape,
                               num_nodes=num_nodes,
                               num_kernel=num_kernel,
                               num_strides=num_strides,
                               latent_dim=latent_dim)
        return model.model
    else:
        print("Wrong model input. model_name should be wither AE or VAE. Exiting...")

##########################################################
# Load the already saved training model
##########################################################

def load_saved_model(model_path):
    """
    Load the saved model from model_path.
    This is useful to load an already trained model
    if ModelCheckpoint is used to save the best model.
    
    At present only can be used to load model for inference.
    Not further further optimization.
    """
    loaded_model = tf.keras.models.load_model(model_path, 
                                              compile=False)
    return loaded_model


##########################################################
# Compile and train the model
##########################################################

def compile_model(model,
            optimizer,
            loss,
            learning_rate=None):
    """
    Compile tensorflow model.
    
    Currently allows 3 optimizers: Adam, RMSprop, SGD
    """
    if learning_rate == 'adam':
        optimizer = optimizers.Adam(learning_rate=int(learning_rate))
    if learning_rate == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=int(learning_rate))
    if learning_rate == 'sgd':
        optimizer = optimizers.SGD(learning_rate=int(learning_rate))

    model.compile(optimizer=optimizer,
                  loss=loss)

def train_model(model,
          train_data,
          epochs,
          batch_size,
          validation_split,
          shuffle,
          callback=False,
          patience=10,
          model_outdir=None):
    """
    Train the tensorflow model
    """
    if callback==True:
        # Define EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   verbose=1)
        # Define ModelCheckpoint
        checkpoint = ModelCheckpoint(model_outdir,
                                     monitor='val_loss',
                                     mode='min',
                                     save_weights_only=False,
                                     save_best_only=True,
                                     verbose=1)

        # Train the model
        history = model.fit(train_data,
                            train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            shuffle=shuffle,
                            callbacks=[early_stop, checkpoint],
                            verbose=1)

        return history

    else:
        # Train the model
        history = model.fit(train_data,
                            train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            shuffle=shuffle,
                            verbose=1)

        return history

##########################################################
# Plot the training loss curves
##########################################################
def plot_train_history(history):
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.close()

