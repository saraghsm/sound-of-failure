##########################################################
# Import default Python libraries
##########################################################
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import tqdm

##########################################################
# Import tensorflow libraries
##########################################################
import tensorflow as tf
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()


##########################################################
# Import sklearn libraries
##########################################################
from sklearn import metrics

##########################################################
# Import custom-made modules
##########################################################
sys.path += ['src/01_data_processing', 'src/02_modelling', ]

import spectrogram as spec
import train_test_split as splt


##########################################################
# Deocode spectrograms and calculate Reconstruction loss
##########################################################

def decode_spectrogram(model, scaler, dim, step, test_file, as_images=True, ndarray=False):
    """
    Reconstruct one spectrogram

    model: Trained autoencoder model
    scaler: Fitted scaler on train
    dim: Time dimension of one spectrogram slice
    step: Sliding window step for creating spectrogram
          chunks
    test_file: Path to the spectrogram
    :return: Original sliced spectrogram and
             Reconstructed sliced spectrogram
    """
    if not ndarray:
        mel = np.load(test_file)
    else:
        mel = test_file
    # Apply scaling to the spectrogram
    scaled_mel = spec.apply_scaler_to_mel(scaler, mel)
    # Create a batch by slicing one spectrogram
    batch = splt.subsample_from_mel(scaled_mel, dim, step, as_images)
    return batch, model.predict(batch)


def reco_loss(model, scaler, dim, step, test_files, as_images=True):
    """
    Calculate reconstruction loss of spectrograms

    model: Trained euoencoder model
    scaler: Fitted scaler on train
    dim: Time dimension of one spectrogram slice
    step: Sliding window step for creating spectrogram
          chunks
    test_file: Path to the spectrogram
    test_labels: Original labels of test_files
    :return: Reconstruction error of all spectrograms
             under test_files by averaging over
             individual spectrogram slices
    """
    # Placeholder for reconstruction loss
    y_pred = np.zeros(len(test_files))

    for idx, test_file in tqdm.tqdm(enumerate(test_files), total=len(test_files)):
        orig_slices, dec_slices = decode_spectrogram(model,
                                                     scaler,
                                                     dim,
                                                     step,
                                                     test_file, as_images)

        # Error calculation
        squared_error = np.square(orig_slices - dec_slices)
        channelwise_error = np.mean(squared_error, axis=-1)
        reconstruction_error = np.mean(channelwise_error)

        y_pred[idx] = reconstruction_error
    return y_pred

def reco_loss_over_time(model, scaler, mel_file, dim, step, as_images=True):
  """
  Calculates the reconstruction error of a mel spectrogram over time. The
  spectrogram is split into smaller time windows (sclices) to which the
  autoencoder model is applied. The mean reconstruction error is assigned to the
  mean time of the respective time window.
  :param model (tensorflow mode): trained autoencoder
  :param scaler (sklearn.preprocessing obj): scaler that is applied
  :param mel_file (str): path to spectrogram
  :param dim (int): dimension of time slices
  :param step (int): step of sliding window
  :as_images (bool): flag to control if spectrograms are processed as images (with one gray scale channel)

  :return: times and reconstruction error by time
  """
  mel = np.load(mel_file)
  # Make times array
  t0 = 0
  t1 = mel.shape[1]
  total_sec = 10
  times = np.arange(t0, t1-dim+step, step)
  times = [min(t, t1-dim) + dim/2 for t in times]
  times = np.array([total_sec*t/t1 for t in times])
  # Make errors array
  orig_slices, dec_slices = decode_spectrogram(model,
                                               scaler,
                                               dim,
                                               step,
                                               mel_file,
                                               as_images)

  squared_error = np.square(orig_slices - dec_slices)
  if as_images:
    squared_error = squared_error[:,:,:,0]
  channelwise_error = np.mean(squared_error, axis=-1)
  timewise_recon_error = np.mean(channelwise_error, axis=-1)

  return times, timewise_recon_error

##########################################################
# Visualize ROC Curve as Classification metrics
##########################################################


def roc_auc_score(y_true, y_pred):
    """
    Area under ROC for test set predictions

    y_true: True labels of test data
    y_pred: Reconstruction error of test data
    :return: ROC AUC SCORE
    """
    auc_score = metrics.roc_auc_score(y_true, y_pred)
    print("Roc AUC score={}".format(auc_score))
    return auc_score


def plot_roc_curve(y_true, y_pred):
    """
    Plot ROC curve for the test predictions

    y_true: True labels of test data
    y_pred: Reconstruction error of test data
    """
    fpr, tpr, thres = metrics.roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linestyle='--',
             label="AUC:{:.2f}".format(auc_score))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.show()
    plt.close()
    return fig

##########################################################
# Encode spectrograms and calculare KL-divergence loss
##########################################################


def encode_spectrogram(model, scaler, dim, step, test_file, direct_encoder=False):
    """
    Encode spectrogram to return the latent space

    model: Trained autoencoder model
    scaler: Fitted scaler on train
    dim: Time dimension of one spectrogram slice
    step: Sliding window step for creating spectrogram
          chunks
    test_file: Path to the spectrogram
    direct_encoder: Whether the given model is an autoencoder
                    or direct encoder
    :return: Encoded representation of the original spectrogram
    """
    mel = np.load(test_file)
    # Apply scaling to the spectrogram
    scaled_mel = spec.apply_scaler_to_mel(scaler, mel)
    # Create a batch by slicing one spectrogram
    batch = splt.subsample_from_mel(scaled_mel, dim, step)

    # Encoded test data in latent space
    if not direct_encoder:
        encoded_data = model.encoder.predict(batch)
    else:
        encoded_data = model.predict(batch)

    return encoded_data


def kl_loss(model, scaler, dim, step, test_files):
    """
    Calculate KL-divergence loss of spectrograms

    model: Trained euoencoder model
    scaler: Fitted scaler on train
    dim: Time dimension of one spectrogram slice
    step: Sliding window step for creating spectrogram
          chunks
    test_file: Path to the spectrogram
    test_labels: Original labels of test_files
    :return: KL-divergence of all spectrograms
             under test_files by averaging over
             individual spectrogram slices
    """
    total_kl_loss = np.zeros(shape=len(test_files))

    for idx, test_file in tqdm.tqdm(enumerate(test_files), total=len(test_files)):
        enc_slices = encode_spectrogram(model,
                                        scaler,
                                        dim, step,
                                        test_file)

        # Mean and log_var for the latent space encodings
        z_mean = enc_slices[0]
        z_log_var = enc_slices[1]

        # kl loss.
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        avg_kl_loss_per_sample = K.mean(kl_loss, axis=0)

        total_kl_loss[idx] = avg_kl_loss_per_sample.eval(
            session=tf.compat.v1.Session())

    return total_kl_loss


##########################################################
# Visualize the separation between train and test samples
##########################################################
def plot_losses(y_true, y_pred, y_train, title, xlabel, y_val=None, thres=None):
    """
    Plot the reco loss or kl-divergence
    for train and test samples.

    y_true: True spectrogram label of test set
    y_pred: Loss for the test set (reco or kl loss)
    y_train: Loss for the train set (reco or kl loss)
    title: Figure title
    xlabel: Title of xlabel
    """
    test_normal = y_pred[np.array(y_true) == 0]
    test_abnormal = y_pred[np.array(y_true) == 1]
    train_normal = y_train
    val_normal = y_val

    labels = ['Train', 'Test (normal)', 'Test (anomaly)']
    colors = ['red', 'green', 'blue']
    datas = [train_normal, test_normal, test_abnormal]

    left_labels = ['Train', 'Validation']
    right_labels = ['Test (normal)', 'Test (anomaly)']
    left_cols = ['red', 'cyan'];
    right_cols = ['green', 'blue']
    left_datas = [train_normal, val_normal]
    right_datas = [test_normal, test_abnormal]

    if (y_val is None) & (thres is None):
        fig, axs = plt.subplots(figsize=(8, 8))
        plt.title(title)
        plt.xlabel(xlabel)
        for data, label, color in zip(datas, labels, colors):
            sns.kdeplot(data,
                        ax=axs,
                        label=label,
                        color=color,
                        alpha=0.5,
                        common_norm=True
                        )
        plt.legend()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        axs[0].set_title("Train and validation data")
        axs[1].set_title("Test data")
        axs[0].set_xlabel(xlabel)
        axs[1].set_xlabel(xlabel)

        for data, label, color in zip(left_datas, left_labels, left_cols):
            sns.distplot(data,
                         ax=axs[0],
                         bins=20,
                         kde=False,
                         label=label,
                         color=color,
                         )
        axs[0].axvline(thres, ymin=0, ymax=1, ls='--', color='black')
        axs[0].annotate('Threshold', xy=(0.048, 80), xytext=(0.048, 80), 
                        weight='bold', fontsize=15)
        axs[0].legend()

        for data, label, color in zip(right_datas, right_labels, right_cols):
            sns.distplot(data,
                         ax=axs[1],
                         bins=20,
                         kde=False,
                         label=label,
                         color=color,
                         )
        axs[1].axvline(thres, ymin=0, ymax=1, ls='--', color='black')
        axs[1].annotate('Threshold', xy=(0.048, 25), xytext=(0.048, 25), 
                        weight='bold', fontsize=15)
        axs[1].legend()

    return fig