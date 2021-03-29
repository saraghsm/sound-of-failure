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
# Deocode spectrograms and calculare Reconstruction loss
##########################################################

def decode_spectrogram(model, scaler, dim, step, test_file, as_images=True):
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
    mel = np.load(test_file)
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
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linestyle='--',
             label="AUC:{:.2f}".format(auc_score))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.show()
    plt.close()

##########################################################
# Encode spectrograms and calculare KL-divergence loss
##########################################################


def encode_spectrogram(model, scaler, dim, step, test_file):
    """
    Encode spectrogram to return the latent space

    model: Trained autoencoder model
    scaler: Fitted scaler on train
    dim: Time dimension of one spectrogram slice
    step: Sliding window step for creating spectrogram
          chunks
    test_file: Path to the spectrogram
    :return: Encoded representation of the original spectrogram
    """
    mel = np.load(test_file)
    # Apply scaling to the spectrogram
    scaled_mel = spec.apply_scaler_to_mel(scaler, mel)
    # Create a batch by slicing one spectrogram
    batch = splt.subsample_from_mel(scaled_mel, dim, step)

    # Encoded test data in latent space
    encoded_data = model.encoder.predict(batch)

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
def plot_losses(y_true, y_pred, y_train, title, xlabel):
    """
    Plot the reco loss or kl-divergence
    for train and test samples.

    y_true: True spectrogram label of test set
    y_pred: Loss for the test set (reco or kl loss)
    y_train: Loss for the train set (reco or kl loss)
    title: Figure title
    xlabel: Title of xlabel
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title(title)

    test_normal = y_pred[np.array(y_true) == 0]
    test_abnormal = y_pred[np.array(y_true) == 1]
    train_normal = y_train

    sns.kdeplot(train_normal,
                ax=ax,
                label='Train',
                color='red',
                alpha=0.5,
                common_norm=True
                )

    sns.kdeplot(test_normal,
                ax=ax,
                label='Test (normal)',
                color='green',
                alpha=0.5,
                common_norm=True)

    sns.kdeplot(test_abnormal,
                ax=ax,
                label='Test (Anomaly)',
                color='blue',
                alpha=0.5,
                common_norm=False)

    plt.legend()
    plt.xlabel(xlabel)
    plt.show()
    plt.close()
