import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import math
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

sys.path.append("../5-Model/")
sys.path.append("../5-Model/version_model")
from rnn_rec import rec_acc
sys.path.append("../4-Dataset")
from dataset_rec import get_datasets

def main():

    data_name='mg_3_rec'
    batch_size=256
    max_len=50

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
    
    print("Use model")
    
    s=0
    n=0

    for i in range(len(test_set)):
        x, y = test_set[i]
        y_hat = model.predict_on_batch(x)
        y=np.squeeze(y)
        y_hat=np.squeeze(y_hat)
        print"================="
        print y.shape
        print y_hat.shape


        for j in range(len(y)):
           
            y_list = y_hat[j] 
            print y_hat[j]
            print y[j]
            max_var = 0
            #for k in range(len(y_list)):
            #    if max_var < y_list[k]:
            #        sec_var = max_var
            #        max_var = y_list[k]
            #print max_var, sec_var

            for k in range(len(y_list)):
                if y_list[k] >=0.5:
                #if y_list[k] >= sec_var:
                    y_list[k] = 1
                else: y_list[k] = 0 
            
            print y_list
            print '==============='

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

      
        for j in range(len(y)): 
            #if (y_hat[j] >0.5): y_predict =1
            #else: y_predict = 0
            #n=n+1 
            print y[j],":", y_hat[j]
            #if (y_predict == y[j]): 
            #    s=s+1
    #per=float(s/n*100)
    #print "Percent: " ,s, "/",n,  per       
            
if __name__ == '__main__':
    main()
