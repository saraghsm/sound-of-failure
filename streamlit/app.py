##########################################################
# Import default Python libraries
##########################################################
import streamlit as st
import sys
import os

##########################################################
# Import custom-made modules
##########################################################
sys.path += ['streamlit/pages', 'src/02_modelling']
import home as home
import exploration as exp
import analysis as ana
import predictions as pred
import train_model_autoencoder as train


##########################################################
# Read data from config
##########################################################
run_id = 'VAE_6dB_valve_id_00_final'
base_conf = train.read_config('conf/conf_base.ini')
BASE_DIR = "./" #base_conf['directories']['base_dir']


##########################################################
# Create the webpage tabs
##########################################################

# Sidebar
sidebar = st.sidebar.radio("Pages", ["Home",
                                     "Exploration",
                                     "Analysis",
                                     "Diagnosis"])

if sidebar == "Home":
    # home.print_home()
    home_header = st.beta_container()
    home_body = st.beta_container()

    # Build home header
    home.build_header(home_header,
                      os.path.join(BASE_DIR, 'streamlit/images/home_image.jpeg'))

    # Build home body
    home.build_body(home_body)


if sidebar == "Exploration":
    #viz.print_home()
    exp_header = st.beta_container()
    exp_body = st.beta_container()

    # Build visualization header
    exp.build_header(exp_header)

    # Build visualization body
    exp.build_body(exp_body)




if sidebar == "Analysis":
    # pred.print_home()
    ana_header = st.beta_container()
    ana_body = st.beta_container()

    # Build prediction header
    ana.build_header(ana_header)

    # Build prediction body
    ana.build_body(ana_body)


if sidebar == "Diagnosis":
    #mod.print_home()
    
    pred_header = st.beta_container()
    pred_body = st.beta_container()

    # Build prediction header
    pred.build_header(pred_header)
    
    # Build prediction body
    pred.build_body(pred_body)
