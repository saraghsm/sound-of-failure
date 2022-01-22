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

**The problem:** Industries experience an average downtime of ~800 hours/year. The average cost of downtime can be as high as ~$20,000 per hour! Often a major cause of downtime is malfunctioning machines. Surveys show that a large number of companies cannot estimate when an equipment starts malfunctioning and only realise when it’s too late. If machine malfunctions can be detected early, downtime costs can be drastically reduced.

**The proposed solution:** The idea is to diagnose machine faillure using their acoustic footprint over time. A machine will produce a different acoustic signature in its abnormal state, compared to its normal state. An algorithm should be able to differentiate between the two sounds.

For a demonstration of the results, feel free to try yourself our [front-end prototype user interface](https://share.streamlit.io/wrijupan/sound-of-failure/main/streamlit/app.py). 

**The dataset:** <!-- Until recently it was not possible to develop solutions outside-in, because there was no publicly available industry data. This has changed in September 2019 when Hitachi, Ltd. released the first of its kind dataset. It contains ca. 100GB of wav files with normal and abnormal sounds of four different types of machines (valves, pumps, fans and slide rails), mixed with their industrial environment background noise. --> A first of its kind, the [MIMII dataset](https://zenodo.org/record/3384388#.YeyHzS8w1-V) was released by Hitachi, Ltd. in September 2019. It contains ca. 100GB of wav files from normal and abnormal sounds of different types of machines, mixed with their industrial environment background noise.

### Model Training

**Data preprocessing:** The sound problem is converted to a computer vision problem by converting the sounds to their image representations, i.e. Mel spectrograms. <!-- The data processing steps include generating Mel spectrograms, standardization, chunking the spectrograms to smaller blocks for generating training and validation data batches. -->

**The Model:** In real situation, only the sound of the normal state of a machin will be available, i.e. the algorithm will not know beforehand how a malfunctioning machine would possibly sound like. Therfore, for training our models we only use the sounds from normally working machines. 

We use an unsupervised Deep Learning approach with Autoencoder architecture for anomaly detection. The Autoencoder is trained to reconstruct back the input sound with a high accuracy (low reconstruction error). Since the model has only been trained on normal sounds of the machin, if an abnormal sound is fed to the trained model as an input, it is not able to reconstruct it well (high reconstruction error). By thresholding on the reconstruction error, we can diagnose a broken machine from its sound.

We have used:
* Convolution Autoencoder
* Variational Autoencoder
* LSTM Autoencoder
<!-- * Convolution LSTM
* Transfer Learning models for feature extraction and using anomaly detection models on the extracted features -->
Variational Autoencoder is our best model in terms of its speed and accuracy.

### Technologies

* Numpy
* Librosa
* Scipy
* Tensorflow, Keras
* Scikit-learn
* etc.

## Project presentation 

If interested, you can checkout [a presentation of the project on YouTube](https://www.youtube.com/watch?v=xaoo_Sy0dwk) for more details and a demo of the results.

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
