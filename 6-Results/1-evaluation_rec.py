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

sys.path.append("../5-Model/version_model")
from rnn_rec import *
sys.path.append("../5-Model")
sys.path.append("../4-Dataset")
from dataset_rec import get_datasets
def rec_acc(y_true, y_pred):
    # y_true: (batch_size, maxlen, 1)
    # y_pred: (batch_size, maxlen, 1)

    # (batch_size, maxlen, 1)
    is_particle_wise_equal = K.equal(y_true, K.round(y_pred))

    # (batch_size, maxlen)
    is_particle_wise_equal = K.squeeze(is_particle_wise_equal, axis=-1)

    # (batch_size, )
    is_jet_wise_correct = K.all(is_particle_wise_equal, axis=1)

    return K.mean(is_jet_wise_correct)


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

def main():

    data_name='mg_3_rec'
    batch_size=128
    max_len=10

    # set name
    data_name = sys.argv[1]
    data_path = '../3-Selector/'+data_name+'/'
    save_path = './'+sys.argv[2]
    #save_path = './ver_6'

    print("Loading dataset.")
    train_set, val_set, test_set = get_datasets(data_path, batch_size, max_len)
    tmp_x, tmp_y = train_set[0]
    x_shape = tmp_x.shape
 
    print("Loading model.")
    model = tf.keras.models.load_model(save_path+'/model.hdf5')
    model.summary()
    
    print("Evaluate")

    #train_y_true, train_y_score = getyinfo(model, train_set)
    test_y_true, test_y_score = getyinfo(model, test_set)
    print test_y_true.shape
    print test_y_score.shape
    print "y true", "  y score"
    for i in range(0,300):
        print test_y_true[i]," :  ", test_y_score[i]
   
if __name__ == '__main__':
    main()
