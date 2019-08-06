#!/usr/bin/python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2" 
import ROOT, sys
import numpy as np
from ROOT   import TLorentzVector
from array  import array
from pprint import pprint

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

sys.path.append("/home/yyoun/deepcmeson/4-Dataset")
from dataset_rec import get_datasets
sys.path.append("/home/yyoun/deepcmeson/5-Model/version_model")
from rnn_rec import *

def main():
    # setting
    if len(sys.argv) == 6:
        data_name = sys.argv[1]
        batch_size = int(sys.argv[2])
        max_len = int(sys.argv[3])
        epochs = int(sys.argv[4])
        model_ver = sys.argv[5]
    else:
        print "Check argv condition"
        sys.exit()

    # set save path
    data_path = '/home/yyoun/deepcmeson/3-Selector/'+data_name+'/'
    save_fold = '/home/yyoun/deepcmeson/6-Results/rec/'
    count_fold= len(os.walk(save_fold).next()[1])+1
    test_ver  = 'test_'+str(count_fold)
    save_path = save_fold+test_ver+'/'
    os.mkdir(save_path)
    
    print test_ver
    
    # save log
    log ='''
[ {test_ver} ]

data name = {data_name}
batch size = {batch_size}
max length = {max_len}
epochs = {epochs}
model = {model_ver}
------------------------------'''.format(test_ver=test_ver, data_name=data_name, batch_size=batch_size,
                                         max_len=max_len, epochs=epochs, model_ver=model_ver)
    with open(save_fold + 'log.txt', 'a') as log_file:
        log_file.write(log)
    with open(save_path + 'log.txt', 'w') as log_file:
        log_file.write(log)

    # set datasets
    train_set, val_set, test_set = get_datasets(data_path, batch_size, max_len)
    tmp_x, tmp_y = train_set[0]
    x_shape = tmp_x.shape
    print 'X Shape: ',tmp_x.shape
    print 'Y Shape: ',tmp_y.shape

    # set model
    if "version" in model_ver:
        model = get_model_fn(model_ver)(x_shape)
    else: 
        model = load_model("/home/yyoun/deepcmeson/6-Results/rec/"+model_ver+"/model.hdf5")

    # save model plot
    print("Save modelplot") 
    keras.utils.plot_model(model, to_file=save_path+'model_plot.png', show_shapes=True, show_layer_names=True)
    
    # set checkpointer and save model
    checkpointer = ModelCheckpoint(filepath=save_path+'model.hdf5', verbose=1, save_best_only=True)
    
    # handle loss step
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

    # save loss and acc
    csv_logger = CSVLogger(save_path + 'CSVLogger.csv')

    # training
    print("Trainig Start") 
    history = model.fit_generator(
        generator = train_set,
        validation_data = val_set,
        steps_per_epoch = len(train_set), 
        epochs = epochs,
        #verbose = 2,
        callbacks = [checkpointer, csv_logger, reduce_lr]
    )
    print("Trainig End") 
   
if __name__ == '__main__':
    main()
