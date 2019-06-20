#!/usr/bin/python
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
import ROOT, sys
import numpy as np
from ROOT   import TLorentzVector
from array  import array
from pprint import pprint

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

sys.path.append("/home/yyoun/deepcmeson/4-Dataset")
from dataset_cfy import get_datasets
sys.path.append("/home/yyoun/deepcmeson/5-Model/version_model")
from rnn_cfy_v1 import build_model

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
    data_name = 'pwg_1_mini'
    epochs = 1
    batch_size = 256
    max_len = 25
         
    if len(sys.argv) == 2:		    
    	data_name = sys.argv[1]
    if len(sys.argv) == 3:
    	data_name = sys.argv[1]
    	epochs = int(sys.argv[2])

    # set save path
    data_path = '/home/yyoun/deepcmeson/3-Selector/'+data_name+'/'
    save_fold = '/home/yyoun/deepcmeson/6-Results/classify/'+data_name
    save_path = save_fold+'/current/'
    if os.path.isdir(save_path):
        print(save_path," is Already exist. Exit.")    
        sys.exit()
    if not os.path.isdir(save_fold):
        os.mkdir(save_fold)
        print("Make ", save_fold)    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # set datasets
    train_set, val_set, test_set = get_datasets(data_path, batch_size, max_len)
    tmp_x, tmp_y = train_set[0]
    x_shape = tmp_x.shape

    # set model
    model = build_model(x_shape)
    
    # save model plot
    print("Save modelplot") 
    keras.utils.plot_model(model, to_file=save_path+'model_plot.png', show_shapes=True, show_layer_names=True)
    
    # set checkpointer, model save
    checkpointer = ModelCheckpoint(filepath=save_path+'model.hdf5', verbose=1, save_best_only=True)
    #model.save(save_path+'rnn_model.h5')

    csv_logger = CSVLogger(save_path + 'CSVLogger.csv')

    # training
    print("Trainig Start") 
    history = model.fit_generator(
        generator = train_set,
        validation_data = val_set,
        steps_per_epoch = len(train_set), 
        epochs = epochs,
        verbose = 2,
        callbacks = [checkpointer, csv_logger]
    )
    print("Trainig End") 
   
    # save loos and acc
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
    train_s_res, train_b_res, test_s_res, test_b_res, test_y_true, test_y_score = evaluate(model, train_set, test_set)

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
