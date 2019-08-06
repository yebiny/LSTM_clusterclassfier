import os
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

sys.path.append("../5-Model")
#from model_cfy import build_model
sys.path.append("../4-Dataset")
from dataset_cfy import get_datasets

def main():

    data_name='mg_3_rec'
    batch_size=256
    max_len=25

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
            if (y_hat[j] >0.5): y_predict =1
            else: y_predict = 0
            n=n+1 
            print y[j],":", y_hat[j],":",y_predict
            if (y_predict == y[j]): 
                s=s+1
    per=float(s/n*100)
    print "Percent: " ,s, "/",n,  per       
            
if __name__ == '__main__':
    main()
