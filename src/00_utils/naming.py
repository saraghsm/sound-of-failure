import os
import sys
import pandas as pd
import shutil
import glob


def make_run_id(model_name, db, machine_type, machine_id):
    """
    Generate default run ID to be used for storing config files, fitted scalers and trained models.
    This function does check if files under this run ID exist.
    :param model_name (str): type of model, takes values 'convAE', 'VAE', 'lstmAE' etc.
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: run ID
    """
    return model_name + '_' + db + '_' + machine_type + '_' + machine_id


def get_conf_path(run_id):
    """
    Generate path for storing/loading configuration file
    :param run_id (str): run ID to be used

    :return: full file path for storing/loading config file
    """
    return os.path.join('conf', run_id + '.ini')


def get_scaler_path(run_id):
    """
    Generate path for storing/loading fitted scaler
    :param run_id (str): run ID to be used

    :return: full file path for storing/loading fitted scaler
    """
    return os.path.join('models', run_id + '.gz')


def get_model_path(run_id):
    """
    Generate path for storing/loading fitted model
    :param run_id (str): run ID to be used

    :return: full file path for storing/loading fitted model
    """
    return os.path.join('models', run_id + '.hdf5')


def show_run_ids():
    """
    Give overview of run IDs under which configuration files, scalers and models
    have been stored in directories conf/ and models/.

    :return: overview data frame
    """
    conf_files = sorted(glob.glob(os.path.join('conf', '*.ini')))
    run_ids = [conf_file.split('/')[-1].replace('.ini','') for conf_file in conf_files]
    run_ids.remove('conf_base')

    df_run_ids = pd.DataFrame({'run_id':run_ids, 'conf': True, 'scaler': False, 'model': False, 'comment':''})
    for index, row in df_run_ids.iterrows():
        run_id = row['run_id']
        df_run_ids.loc[index, 'scaler'] = os.path.exists(get_scaler_path(run_id))
        df_run_ids.loc[index, 'model'] = os.path.exists(get_model_path(run_id))
        if len(run_id.split('_')) == 5:
            df_run_ids.loc[index, 'comment'] = 'Default run ID. Archive run to prevent overwriting.'

    return df_run_ids


def copy_run(run_id, new_run_id):
    """
    Copy all files stored under a given run ID and store under a new run ID.
    """
    for fct in [get_conf_path, get_scaler_path, get_model_path]:
        path = fct(run_id)
        if os.path.exists(path):
            new_path = fct(new_run_id)
            shutil.copy(path, new_path)
            print(f'Copy {path} to {new_path}')


def delete_run(run_id):
    """
    Delete all files stored under a given run ID.
    """
    print(f'Delete run {run_id}.')
    for fct in [get_conf_path, get_scaler_path, get_model_path]:
        path = fct(run_id)
        if os.path.exists(path):
            os.remove(path)
            print(f'{path} deleted.')


def get_new_run_id(run_id):
    """
    Generate a new run ID by appending '_<count>' to a given run ID.
    """
    count = 1
    new_run_id = run_id + '_' + str(count)

    while (os.path.exists(get_scaler_path(new_run_id)) or
           os.path.exists(get_model_path(new_run_id)) or
           os.path.exists(get_conf_path(new_run_id))):
        count += 1
        new_run_id = run_id + '_' + str(count)

    return new_run_id


def archive_run(run_id):
    """
    Copy all files stored under a given run ID and store under a newly
    created run ID.
    """
    print(f'Archive run {run_id}.')
    new_run_id = get_new_run_id(run_id)
    copy_run(run_id, new_run_id)