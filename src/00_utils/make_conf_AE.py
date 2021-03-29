from configparser import ConfigParser
import argparse
import os
import sys

sys.path += ['src/00_utils']
from naming import *

# Config parser
config = ConfigParser()

# Argument parser
parser = argparse.ArgumentParser(
    description="Params for Spectrogram and Autoencoder (AE, VAE or lstmAE)")

# Add argument through the command line
parser.add_argument('-prj', '--project', metavar='_',
                    required=True, help='directory hat holds sound-of-failure')
parser.add_argument('-ae', '--ae', type=str, metavar='', default='AE',
                    help='Type of Autoencoder (AE, VAE or lstmAE)')
parser.add_argument('-noise', '--noise', type=str, metavar='', default='6dB',
                    help='Level of background noise (6dB, 0dB or min6dB)')
parser.add_argument('-type', '--type', type=str, metavar='',
                    help='Type of machine (valve, slider, pump or fan)')
parser.add_argument('-id', '--id', type=str, metavar='',
                    help='Machine ID (e.g. id_00)')
parser.add_argument('-mel', '--n_mels', type=int,
                    metavar='', default=128, help='No. of mel bands')
parser.add_argument('-fft', '--n_fft', type=int,
                    metavar='', default=1024, help='No. of FFT bands')
parser.add_argument('-hop', '--hop_length', type=int, metavar='',
                    default=512, help='Hop length for FFT calc')
parser.add_argument('-dim', '--dim', type=int, metavar='',
                    default=32, help='Time dimension of Spectrogram block')
parser.add_argument('-s', '--step', type=int, metavar='',
                    default=8, help='Sliding window step for Spectrogram chunking')
args = parser.parse_args()

# Mel spectrogram parameters
config['melspec'] = {
    'n_mels': args.n_mels,
    'n_fft': args.n_fft,
    'hop_length': args.hop_length,
    'dim': args.dim,
    'step': args.step,
    'power': 2.0,
    'window': 'hann',
}

# Parameters for accessing the data
config['data'] = {
    'noise': args.noise,
    'machine': args.type,
    'machine_id': args.id
}

# Parameters for model training
config['model'] = {
    'scaler': 'StandardScaler',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'epochs': 100,
    'batch_size': 64,
    'val_split': 0.1,
    'shuffle': True
}

# Parameters for the Convolutional Autoencoder
if args.ae in ['AE', 'VAE']:
    config['autoencoder'] = {
        'model_name': args.ae,
        'latentdim': 20,
        'num_nodes': [32, 64, 128, 256,],
        'num_kernel': [5, 5, 3, 3,],
        'num_strides': [(1, 2), (2, 2), (2, 2), (1, 2),]
    }

if args.ae == 'lstmAE':
    config['autoencoder'] = {
        'model_name': args.ae,
        'num_nodes': [128, 128, 128,],
    }

if __name__ == '__main__':
    model_name = args.ae
    db = args.noise
    machine_type = args.type
    machine_id = args.id
    run_id = make_run_id(model_name, db, machine_type, machine_id)
    if args.ae in ['AE', 'VAE', 'lstmAE']:
        conf_path = get_conf_path(run_id)
    else:
        print("Wrong input. '-ae' should be one of 'AE', 'VAE' or 'lstmAE'")

    with open(conf_path, 'w') as AE_config:
        config.write(AE_config)
