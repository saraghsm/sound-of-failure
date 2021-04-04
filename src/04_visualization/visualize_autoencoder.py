##########################################################
# Import Python libraries libraries
##########################################################
import sys
import tqdm
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from librosa import display as ld


##########################################################
# Import Machine Learning libraries
##########################################################
import tensorflow as tf
from tensorflow.keras.models import Model
tf.compat.v1.disable_eager_execution()
from sklearn import manifold

##########################################################
# Import custom-made modules
##########################################################
sys.path += ['src/01_data_processing',  'src/03_modell_evaluation',]

import spectrogram as spec
import eval_model_autoencoder as eval


##########################################################
# Load encoder and decoder from saved autoencoder model
##########################################################
def load_encoder_decoder(saved_model):
    """
    Extract the encoder and decoder components
    from the saved model

    saved_model: Tensorflow model obtained from
                    the ModelCheckPoint
    """
    # Load encoder
    encoder_input = saved_model.layers[1].input
    encoder_output = saved_model.layers[1].output
    encoder = Model(encoder_input, encoder_output)

    # Load decoder
    decoder_input = saved_model.layers[2].input
    decoder_output = saved_model.layers[2].output
    decoder = Model(decoder_input, decoder_output)

    return encoder, decoder


##########################################################
# Render the spectrograms into latent space encodings
##########################################################

def encode_test_files(model,
                      scaler,
                      dim,
                      step,
                      spectrograms):
    """
    Encode input spectrograms into latent space
    and output the mean and log_variance from the
    encodings.
    """
    for idx, spectrogram in tqdm.tqdm(enumerate(spectrograms),
                                      total=len(spectrograms)):

        encoded_ = eval.encode_spectrogram(model=model,
                                           scaler=scaler,
                                           dim=dim,
                                           step=step,
                                           test_file=spectrogram,
                                           direct_encoder=True)
        z_mean = encoded_[0]
        z_logvar = encoded_[1]

        if idx == 0:
            encoded_test_mean = z_mean
            encoded_test_logvar = z_logvar
        else:
            encoded_test_mean = np.concatenate((encoded_test_mean, z_mean))
            encoded_test_logvar = np.concatenate((encoded_test_logvar, z_logvar))
    return encoded_test_mean, encoded_test_logvar

##########################################################
# Plotting functions
##########################################################

def plot_spectrogram(spec_file, sr, ndarray=False):
    """
    Plot a spectrogram from the 'spec_file'
    having a given sampling rate 'sr'
    """
    fig, ax = plt.subplots()
    if not ndarray:
        spectrogram = np.load(spec_file)
    else:
        spectrogram = spec_file

    img = ld.specshow(spectrogram,
                      x_axis='time',
                      y_axis='mel',
                      sr=sr,
                      fmax=0.5 * sr,
                      ax=ax)

    fig.colorbar(img, ax=ax)
    #plt.show()
    #plt.close()
    return fig


def render_predictions(original_spec, reconstructed_spec, sr):
    """
    Given the original and reconstructed spectrogram slices
    (the slices come by subsampling a bigger spectrogram),
    plot them one below the other.
    """
    fig, axs = plt.subplots(3, 5, figsize=(9, 4))
    

    for index in np.arange(5, 10):
        image_in = original_spec[index].reshape(original_spec[0].shape[0],
                                                original_spec[0].shape[1])

        image_out = reconstructed_spec[index].reshape(reconstructed_spec[0].shape[0],
                                                      reconstructed_spec[0].shape[1])
                                                      
        reco_error = np.square(image_in - image_out)
        index -= 5
                                                      

        orig = ld.specshow(image_in.T,
                           x_axis='time',
                           y_axis='mel',
                           sr=sr,
                           fmax=sr * 0.5,
                           ax=axs[0, index],
                           cmap='inferno')

        reco = ld.specshow(image_out.T,
                           x_axis='time',
                           y_axis='mel',
                           sr=sr,
                           fmax=sr * 0.5,
                           ax=axs[1, index],
                           cmap='inferno')
                           
        err = ld.specshow(reco_error.T,
                           x_axis='time',
                           y_axis='mel',
                           sr=sr,
                           fmax=sr * 0.5,
                           ax=axs[2, index],
                           cmap='inferno')

        axs[0, index].set_title("Original", fontsize=8)
        axs[0, index].axis('off')
        axs[1, index].set_title("Reconstructed", fontsize=8)
        axs[1, index].axis('off')
        axs[2, index].set_title("Squared error", fontsize=8)
        axs[2, index].axis('off')

    #plt.show()
    #plt.close()
    return fig


def Tsne_Projection_Of_Latent(data, data_label):
  """
  Perform dimensionality reduction on the latent spae
  to reduce it to 2 dimensions using TSNE and visualize
  the latent space encodings.
  """

  tsne_latent = manifold.TSNE(n_components=2, init='pca', random_state=0)

  X_tsne = tsne_latent.fit_transform(data)

  fig = plt.figure(figsize=(6,6))
  plt.scatter(X_tsne[:, 0], X_tsne[:, 1],  c=data_label, cmap='inferno')
  #sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=data_label, cmap="inferno")
  plt.colorbar()
  plt.show()
  plt.close()
  return fig


