import os
import ROOT, sys
#from tensorflow.keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model
import numpy as np
from numpy import argmax
from array import array

sys.path.append("../5-Model")
from model_cfy import build_model
sys.path.append("../4-Dataset")
from dataset_cfy import get_datasets

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

    #default
    batch_size = 256
    max_len = 15

    # set name
    data_name = sys.argv[1]
    data_path = '../3-Selector/'+data_name+'/'
    save_path = './'+sys.argv[2]

    print("Loading dataset.")
    train_set, val_set, test_set = get_datasets(data_path, batch_size, max_len)
    tmp_x, tmp_y = train_set[0]
    x_shape = tmp_x.shape
 
    print("Loading model.")
    model = tf.keras.models.load_model(save_path+'/rnn_model.h5')
    model.load_weights(save_path+'/weights.hdf5')
    model.summary()
    
    print("Evaluate")
    train_s_res, train_b_res, test_s_res, test_b_res, test_y_true, test_y_score = evaluate(model, train_set, test_set)
    
    #y_vloss = history.history['val_loss']
    #y_loss = history.history['loss']
    #y_acc = history.history['acc']   
    #y_vacc = history.history['val_acc']   
    
    print("Save results")
    np.savez(
        save_path+'/info_eval.npz',
        # learning curve
        #y_vloss = y_vloss,
        #y_loss = y_loss,
        #y_acc = y_acc,
        #y_vacc = y_vacc,
        # responce
        train_sig_response = train_s_res,
        train_bkg_response = train_b_res,
        test_sig_response = test_s_res,
        test_bkg_response = test_b_res,
        # roc curve
        test_y_true = test_y_true,
        test_y_score=test_y_score,
    )
 
if __name__ == '__main__':
    main()
