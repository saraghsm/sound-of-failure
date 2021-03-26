import os

def get_default_name_string(model_name, db, machine_type, machine_id):
    """
    Generate default name string to be used for storing config files, fitted scalers and trained models.
    This function does check if files using this name exist.
    :param model_name (str): type of model, takes values 'convAE', 'VAE', 'lstmAE' etc.
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: name string
    """
    return model_name + '_' + db + '_' + machine_type + '_' + machine_id


def get_conf_path(name_string):
    """
    Generate path for storing configuration file
    :param name_string (str): name string to be used

    :return: full file path for storing config file
    """
    return os.path.join('conf', name_string + '.ini')


def get_scaler_path(name_string):
    """
    Generate path for storing fitted scaler
    :param name_string (str): name string to be used

    :return: full file path for storing fitted scaler
    """
    return os.path.join('models', name_string + '.gz')


def get_model_path(name_string):
    """
    Generate path for storing fitted scaler
    :param name_string (str): name string to be used

    :return: full file path for storing fitted scaler
    """
    return os.path.join('models', name_string + '.hdf5')


def get_new_name_string(model_name, db, machine_type, machine_id):
    """
    Generate default name string to be used for storing config files, fitted scalers and trained models.
    This function checks if files using this name exist and appends a counter to create a new unique name string.
    :param model_name (str): type of model, takes values 'convAE', 'VAE', 'lstmAE' etc.
    :param db (str): noise level, takes values '6dB', '0dB' or 'min6dB'
    :param machine_type (str): type of machine, takes values 'fan', 'pump', 'slider', 'valve'
    :param machine_id (str): id of machine, takes values 'id_00', 'id_02' etc.

    :return: name string
    """
    name_string = get_default_name_string(model_name, db, machine_type, machine_id)
    count = 1
    new_name_string = name_string

    while os.path.exists(get_scaler_path(new_name_string)):
        new_name_string = name_string + '_' + str(count)
        count += 1

    while os.path.exists(get_model_path(new_name_string)):
        new_name_string = name_string + '_' + str(count)
        count += 1

    while os.path.exists(get_conf_path(new_name_string)):
        new_name_string = name_string + '_' + str(count)
        count += 1

    return new_name_string
