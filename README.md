Sound of failure
==============================

An AI solution to reduce industrial downtime by diagnosing the failure of machines using their acoustic footprint.

#### -- Project Status: [Active]

### Collaborators
|Name     |  Github Page   |
|---------|-----------------|
| Wrijupan Bhattacharyya | [wrijupan](https://github.com/wrijupan)|
| Sara Ghasemi | [saraghsm](https://github.com/saraghsm) |
| Niklas Hübel | [NikoHobel](https://github.com/NikoHobel) |

## Project Description

**The problem:** Industries experience an average downtime of ~800 hours/year. The average cost of downtime can be as high as ~$20,000 per hour! Often a major cause of downtime is malfunctioning machines. During downtime, the overhead operating costs keeps growing without a significant increase in productivity. A survey in 2017 had found that 70% of companies cannot estimate when an equipment starts malfunctioning and only realise when it’s too late. If malfunctions can be detected early, downtime costs can be drastically reduced.

**The proposed solution:** The idea is to diagnose machine faillure using their acoustic footprint over time. A machine will produce a different acoustic signature in its abnormal state compared to its normal state. An algorithm should be able to differentiate between the two sounds.

**The dataset:** Until recently it was not possible to develop solutions outside-in, because there was no publicly available industry data. This has changed in September 2019 when Hitachi, Ltd. released the first of its kind dataset. It contains ca. 100GB of wav files with normal and abnormal sounds of four different types of machines (valves, pumps, fans and slide rails), mixed with their industrial environment background noise. 

### Methods Used

**Data preprocessing:** The sound problem is converted to a computer vision problem by converted the sound to its image representation (i.e. Mel spectrograms). The data processing steps include generating Mel spectrograms, standardization, chunking the spectrograms to smaller blocks for generating training and validation data batches.

**Machine Learning:** In general, only machine sounds from a normal state of an instrument will be available, i.e. the algorithm will not know beforehand how an abnormal sound looks like. So the training would be unsupervised using only normal sound data. Then during validation when the algorithm encounters an abnormal sound, it will identify that as an outlier.

The approaches that is used are (under active development, the results will appear very very soon here!):
* Convolution Autoencoder
* Variational Autoencoder
* LSTM
* Convolution LSTM
* Transfer Learning models for feature extraction and using anomaly detection models on the extracted features

### Technologies

* Numpy
* Librosa
* Scipy
* Tensorflow, Keras
* Scikit-learn
* etc.

## Getting Started

0. Get the raw data from... (TBC)

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. cd sound-of-failure; mkdir data
3. ...
4. ...

Project Organization
------------

    ├── LICENSE
    ├── Makefile                 <- Makefile with commands like `make data` or `make train` (To be added)
    ├── README.md                <- The top-level README for description of this project and how to use it.
    ├── requirements.txt         <- The requirements file for reproducing the analysis environment
    ├── setup.py                 <- makes project pip installable (pip install -e .) so src can be imported (To be added)
    │
    ├── conf                     <- (Directory) Configuration files
    │
    ├── data                     <- (Directory) Processed data
    │   └── mel_spectrograms     <- Un-scaled Mel Spectrograms, 1 per audio file
    │
    ├── docs                     <- (Directory) A default Sphinx project; see sphinx-doc.org for details (TBC)
    │
    ├── models                   <- (Directory) Trained and serialized models
    │
    ├── notebooks                <- (Directory) Jupyter notebooks. Naming convention is the creator's initials,
    │                               a number for ordering (typically the date), and a short `-` delimited description.
    │
    ├── references               <- (Directory) Reading material can be saved here.
    │
    ├── outputs                  <- (Directory) Model outputs, material for reporting (e.g. tabels, histograms, etc.)
    │   └── figures              <- Generated graphics and figures to be used in reporting
    │
    ├── TEST                     <- (Directory) Used for private experimentation
    |
    └── src                      <- (Directory) Source code for use in this project.
        ├── __init__.py          <- Makes src a Python module
        │
        ├── 00_utils             <- Functions used across the project
        │
        ├── 01_data_processing   <- Scripts to turn raw data into features for modeling
        │
        ├── 02_modelling         <- Scripts to train models and then use trained models to make predictions
        │
        ├── 03_modell_evaluation <- Scripts that analyse model performance and model selection
        │
        └── 04_visualization     <- Scripts to create exploratory and results oriented visualizations
    
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
