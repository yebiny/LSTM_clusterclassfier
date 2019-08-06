#!/usr/bin/python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5" 
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
from dataset_cfy import get_datasets
sys.path.append("/home/yyoun/deepcmeson/5-Model/version_model")
from rnn_cfy import *

def getyinfo(model, xset):
    
    y_true = []
    y_score = []
    for idx in range(len(xset)):
        x, y = xset[idx]
        y_predict = model.predict_on_batch(x)
        y_true.append(y)
        y_score.append(y_predict)
   
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    
    return y_true, y_score

def evaluate(model, train_set, test_set):
    
    train_y_true, train_y_score = getyinfo(model, train_set)
    test_y_true, test_y_score = getyinfo(model, test_set)
    
    train_is_sig = train_y_true.astype(np.bool)
    train_is_bkg = np.logical_not(train_is_sig)
    test_is_sig = test_y_true.astype(np.bool)
    test_is_bkg = np.logical_not(test_is_sig)

    train_sig_response = train_y_score[train_is_sig]
    train_bkg_response = train_y_score[train_is_bkg]
    test_sig_response = test_y_score[test_is_sig]
    test_bkg_response = test_y_score[test_is_bkg] 

    return train_sig_response, train_bkg_response, test_sig_response, test_bkg_response, test_y_true, test_y_score

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
    save_fold = '/home/yyoun/deepcmeson/6-Results/cfy/'
    count_fold= len(os.walk(save_fold).next()[1])+1
    test_ver  = 'test_'+str(count_fold)
    save_path = save_fold+test_ver+'/'
    os.mkdir(save_path)
    
    print 'Save folder: ',test_ver
    
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
        model = load_model("/home/yyoun/deepcmeson/6-Results/cfy/"+model_ver+"/model.hdf5")
        #model = load_model("/home/yyoun/deepcmeson/6-Results/bfres/classify/pwg_1_full/v5_ep30/rnn_model.h5")

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
   
    # save loss and acc for draw
    print("Save loss and acc") 
    y_loss = history.history['loss']
    y_acc = history.history['acc']   
    y_vloss = history.history['val_loss']
    y_vacc = history.history['val_acc']   
    np.savez(
        save_path+'info_learning.npz',
        y_loss = y_loss,
        y_acc = y_acc,
        y_vloss = y_vloss,
        y_vacc = y_vacc
    )

    # evaluation
    print("Evaluation") 
    train_s_res, train_b_res, test_s_res , test_b_res , test_y_true, test_y_score = evaluate(model, train_set, test_set)

    # save evaluation results
    print("Save results") 
    np.savez(
        save_path+'info_eval.npz',
        # roc curve
        test_y_true = test_y_true,
        test_y_score=test_y_score,
        # responce 
        train_sig_response=train_s_res,
        train_bkg_response=train_b_res,
        test_sig_response = test_s_res,
        test_bkg_response = test_b_res,
    )

if __name__ == '__main__':
    main()
