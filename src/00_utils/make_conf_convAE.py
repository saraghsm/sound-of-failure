from configparser import ConfigParser
import argparse

# Config parser
config = ConfigParser()

# Argument parser
parser = argparse.ArgumentParser(
    description="Params for Spectrogram and Conv AE")

# Add argument through the command line
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
    'noise': '6dB',
    'machine': 'valve',
    'machine_id': 'id_00'
}

# Parameters for model training
config['model'] = {
    'scaler': 'StandardScaler',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'epochs': 30,
    'batch_size': 64,
    'val_split': 0.1,
    'shuffle': True
}

# Parameters for the Convolutional Autoencoder
config['autoencoder'] = {
    'latentdim': 20,
    'num_nodes': [32, 64, 128, 256,],
    'num_kernel': [5, 5, 3, 3,],
    'num_strides': [(1, 2), (2, 2), (2, 2), (1, 2),]
}

if __name__ == '__main__':
    config_filename = '/gdrive/MyDrive/sound-of-failure/conf/conf_convAE.ini'

    with open(config_filename, 'w') as convAE_config:
        config.write(convAE_config)
