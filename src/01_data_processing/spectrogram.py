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

sys.path += ['../src/filecheck', '../src/preprocessing',]
from flatfielding import *
from filepath import *


def mel_spectrogram(file, 
                    scaler, 
                    n_fft,
                    n_mels, 
                    hop_length, 
                    window, 
                    power, 
                    dim, 
                    step):
    """
    Calculates spectrogram of the given audio
    :param file (str): path to wav
    :param scaler (sklearn.preprocessing obj)
    :param n_fft (int): no.of samples in each frame
    :param hop_length (int): hop samples
    :param n_mels (int): no. of mel-bands
    :param power (int): 1 for energy, 2 for power
    :param window (str): 'STFT' window, e.g. 'Hann'
    :dim (int): dimension of time slices
    :step (int): step of sliding window

    :return: Mel spectrogram from sliding window 
                (no. of slices. dim, n_mels)
    """
    y, sr = librosa.load(file, sr=None, mono=True)

    mel_spec = librosa.feature.melspectrogram(y,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              sr=sr,
                                              n_mels=n_mels,
                                              window=window,
                                              power=power)

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    #log_mel_spec = 20.0 / power * np.log10(mel_spec + sys.float_info.epsilon)
    
    # Transform with scaler
    log_mel_spec = scaler.transform(log_mel_spec)
    
    log_mel_spec = log_mel_spec.T
    
    length = log_mel_spec.shape[0]
    
    start_indices = np.arange(length - dim + step, step=step)
    
    for idx in range(len(start_indices)):
        start = min(start_indices[idx], length - dim)
    
        one_slice = log_mel_spec[start : start + dim, :]
        one_slice = one_slice.reshape((1, one_slice.shape[0], one_slice.shape[1]))
    
        if idx == 0:
          batch = one_slice
        else:
          batch = np.concatenate((batch, one_slice))

    return batch
    
    
def fit_scaler(filelist, n_fft, 
               n_mels, hop_length, 
               window, power, 
               scaler_dir,
               scaler=StandardScaler()):
    """
    Function for fitting a Normalizer (sklearn.preprocessing)
    """
    train_path = '/'.join(filelist[0].split('/')[:-1])
    print("Fitting {} to train data in {}".format(scaler, train_path))

  
    for i, file in tqdm.tqdm(enumerate(filelist), total=len(filelist)):
    
        y, sr = librosa.load(file, sr=None, mono=True)
    
        mel_spectrogram = librosa.feature.melspectrogram(y, 
                                                        n_fft=n_fft,
                                                        n_mels=n_mels,
                                                        hop_length=hop_length,
                                                        window=window,
                                                        power=power)
        
        #log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, 
                                                  ref=np.max)
    
        scaler.partial_fit(log_mel_spectrogram)
        
        if i==0:
                dirpath = '/'.join(file.split("/")[:-2])
                scalerpath = dirpath + '/' + scaler_dir
                
                if not os.path.exists(scalerpath):
                    os.mkdir(scalerpath)
                else:
                    shutil.rmtree(scalerpath)
                    os.mkdir(scalerpath)
                    
    dump(scaler, scalerpath+ "/" + "scaler.bin", compress=True)
    
    return scaler


def mel_spectrogram_list(filelist, 
                         out_dir,
                         scaler_dir,
                         n_fft, 
                         n_mels, 
                         hop_length, 
                         window, 
                         power, 
                         dim, 
                         step):
    """
    Generate mel spectrograms for all the
    Audio files in 'filelist'.
    
    Before running this run 'fit_scaler'
    to store or crate the scaler
    
    :param filepath (list): Path to all .wav files
        Output from 'extract_filepath' func.
    :param out_dir (str): path to store the features
    :param scaler_dir (str): name of directory where
                            Scaler is stored.
    
    :return: Feature vector (no.of.samples, dim, n_mels)
        (Only for visualization. All features stored in 'out_dir')
    """
    for i, file in tqdm.tqdm(enumerate(filelist), total=len(filelist)):
        
        if i==0:
                    dirpath = '/'.join(file.split("/")[:-2])
                    outpath = dirpath + '/' + out_dir
                    scalerpath = dirpath + '/' + scaler_dir
                    
                    if not os.path.exists(outpath):
                        os.mkdir(outpath)
                    else:
                        shutil.rmtree(outpath)
                        os.mkdir(outpath)
                        
                    if not os.path.exists(scalerpath):
                        print("Path to scaler {} does not exist"
                              .format(scalerpath))
                        print("First set 'fit_scaler=True' and rerun...")
                        sys.exit() 
                        
        
                    scaler = load(scalerpath+ "/" + "scaler.bin")
        
        

        block = mel_spectrogram(file=file, scaler=scaler, 
                                  n_fft=n_fft, n_mels=n_mels, 
                                  hop_length=hop_length, 
                                  window=window, power=power, 
                                  dim=dim, 
                                  step=step)
        
        if i == 0:
          vectorarray = block
        else:
          vectorarray = np.concatenate((vectorarray, block))

    vectorarray = vectorarray.reshape((vectorarray.shape[0], 
                              vectorarray.shape[1], 
                              vectorarray.shape[2], 
                              1
                              ))
                              
    # Save features
    np.save(outpath+'/'+'data.npy', vectorarray)
    
    return vectorarray
    