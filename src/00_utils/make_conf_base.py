import os
from configparser import ConfigParser
import argparse

# Config parser
config_base = ConfigParser()

# Argument parser
parser = argparse.ArgumentParser(
    description="Params for Spectrogram and Conv AE")

# Add argument through the command line
parser.add_argument('-raw', '--raw_data', metavar='',
                    required=True, help='Path to raw data inside gdrive/MyDrive')
parser.add_argument('-prj', '--project', metavar='',
                    required=True, help='directory that holds sound-of-failure')
args = parser.parse_args()

# Build the directory configuration
config_base['directories'] = {
    'raw_data_dir': args.raw_data,
    'base_dir': os.path.join(args.project, 'sound-of-failure')
}

if __name__ == '__main__':
    config_filename = os.path.join(args.project, 'sound-of-failure/conf/conf_base.ini')

    with open(config_filename, 'w') as base_file:
        config_base.write(base_file)
