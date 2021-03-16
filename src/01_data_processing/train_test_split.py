import os
import shutil
import sys
import random
import numpy as np

sys.path += ['../src/filecheck', '../src/preprocessing',]
from flatfielding import *
from filepath import *


def rename_wav(data_dir, 
               machine_types, 
               machine_ids, 
               ):
    """
    Renames wav files.
        Appends 'normal_' before the filename
        of normal sounds and appends 'abnormal_'
        before filename of abnormal sounds.
        
    data_dir (str): path to the data directory
        Inside the data_dir expects the following:
        Path_to_data/<machine>/<machine_ids>/<normal 'or' abnormal>
        
    machine_types (list): ["valves","fan" etc]
    
    machine_ids (list): ["id_00", "id_02" etc]
    """
    for m_type in machine_types:
        
        for m_id in machine_ids:
            
            norm = extract_filepath(data_dir, 
                                    inst=m_type, 
                                    id=m_id, 
                                    cond='normal')
            
            abnorm = extract_filepath(data_dir, 
                                    inst=m_type, 
                                    id=m_id, 
                                    cond='abnormal')
            
            rename_norm = ['/'.join(wav.split('/')[:-1])+'/normal_'
                           + (wav.split('/')[-1]).split('_')[-1] for wav in norm]
            
            rename_abnorm = ['/'.join(wav.split('/')[:-1])+'/abnormal_'
                           + (wav.split('/')[-1]).split('_')[-1] for wav in abnorm]
            
            
            for orig_norm, renam_norm in zip(norm, rename_norm):
                if not 'normal' in orig_norm:
                    shutil.move(orig_norm, renam_norm)
                else:
                    print("File {} already renamed. Skipping...".format(renam_norm))
                    pass
                
            for orig_abnorm, renam_abnorm in zip(abnorm, rename_abnorm):
                if not 'abnormal' in orig_abnorm:
                    shutil.move(orig_abnorm, renam_abnorm)
                else:
                    print("File {} already renamed. Skipping...".format(renam_abnorm))
                    pass
                
  
def train_test_split(data_dir, 
                     machine_types, 
                     machine_ids, 
                     dir_types):
    """
    Splits the train and test samples.
    Creates "train" and "test" dirs under each
    <data_dir>/<machine_type>/<machine_id>.
    
    Test data- All abnormal + same no. of randomly
                selected normals.
    Train data- Rest of the normals which are not
                part of the test set.
                
    data_dir (str): path to the data directory
        Inside the data_dir expects the following:
        Path_to_data/<machine>/<machine_ids>/<normal 'or' abnormal>
        
    machine_types (list): ["valves","fan" etc]
    
    machine_ids (list): ["id_00", "id_02" etc]
    
    dir_types (list): ["train", "test"]
    """
    
    # Loop for creating the test and train dir
    for m_type in machine_types:
        
        for m_id in machine_ids:
            
            for d_type in dir_types:
                
                create_dir = '/'.join([data_dir, m_type, m_id, d_type])
                
                if not os.path.exists(create_dir):
                    os.mkdir(create_dir)
                    
    # Loop for copying data to train and test
    for m_type in machine_types:
        
        for m_id in machine_ids:
            
            train_dir = '/'.join([data_dir, m_type, m_id, 'train'])
            test_dir = '/'.join([data_dir, m_type, m_id, 'test'])
            
            #List of abnormal files
            abnormal_filepath = extract_filepath(data_dir, 
                                                inst=m_type, 
                                                id=m_id, 
                                                cond='abnormal')
            # Number of abnormal files
            abnormal_no = len(abnormal_filepath)
            
            #List of normal files
            normal_filepath = extract_filepath(data_dir, 
                                                inst=m_type, 
                                                id=m_id, 
                                                cond='normal')
            
            # Randomly select normal files = abnormal file no
            normal_selected = random.sample(normal_filepath, 
                                            abnormal_no)
            
            # Move files to the test directory
            [shutil.move(normal_test, test_dir) 
            for normal_test in normal_selected]
                        
            [shutil.move(abnormal_test, test_dir) 
            for abnormal_test in abnormal_filepath]
            
            # Move rest of normal to train dir
            normal_remaining = extract_filepath(data_dir, 
                                                inst=m_type, 
                                                id=m_id, 
                                                cond='normal')
            
            [shutil.move(normal_train, train_dir) 
             for normal_train in normal_remaining]
            

            
            
def undo_train_test(data_dir, 
                     machine_types, 
                     machine_ids, 
                     dir_types):
    """
    Undo the train-test split.
    Moves back the wav files from the 'train' and 'test'
    back to 'normal' or 'abnormal'.
                
    data_dir (str): path to the data directory
        Inside the data_dir expects the following:
        Path_to_data/<machine>/<machine_ids>/<normal 'or' abnormal>
        
    machine_types (list): ["valves","fan" etc]
    
    machine_ids (list): ["id_00", "id_02" etc]
    
    dir_types (list): ["train", "test"]
    """
    
    for m_type in machine_types:
        
        for m_id in machine_ids:
            norm_path = '/'.join([data_dir, m_type, m_id, 'normal'])
            abnorm_path = '/'.join([data_dir, m_type, m_id, 'abnormal'])
            
            for d_type in dir_types:
                
                filelist = extract_filepath(data_dir, 
                                            inst=m_type, 
                                            id=m_id, 
                                            cond=d_type)
                
                if d_type == 'train':
                    
                    [shutil.move(f, norm_path) for f in filelist 
                     if not 'ab' in f]
                    
                elif d_type == 'test':
                    
                    [shutil.move(f, norm_path) for f in filelist 
                     if not 'ab' in f]
                    
                    [shutil.move(f, abnorm_path) for f in filelist 
                     if 'ab' in f]
