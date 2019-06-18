import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2" 
import ROOT, sys
from ROOT import TLorentzVector
from array import array
import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Bidirectional, Dropout
from tensorflow.keras.utils import Sequence, plot_model 
from tensorflow.keras.callbacks import ModelCheckpoint
if tf.test.is_gpu_available(cuda_only=True):
    from tensorflow.keras.layers import CuDNNLSTM as LSTM
else:
    from tensorflow.keras.layers import LSTM
from sklearn.utils.class_weight import compute_class_weight
from pprint import pprint

sys.path.append("../4-Dataset")
from dataset_rec import get_datasets

''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''                                                '''
'''   python3 model.py [title] [number of epochs]  '''
'''                                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''

def build_model(x_shape,max_len):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(x_shape[1],x_shape[2])))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation='sigmoid'))
    model.add(Dense(max_len, activation='sigmoid'))
    model.compile('adam',loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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

    # set name
    modelname = 'test'
    epochs = 10
    batch_size = 256
    max_len = 5
    
    if len(sys.argv) == 2:          
        modelname = sys.argv[1]
    if len(sys.argv) == 3:
        modelname = sys.argv[1]
        epochs = int(sys.argv[2])

    # set save path
    folder_path = '../3-Selector/'+modelname+'/'
    save_path = '../6-Results/reconstruction/'+modelname+'/'
    if os.path.isdir(save_path):
        print("Already exist. Exit.")    
        sys.exit()
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # set datasets
    train_set, val_set, test_set = get_datasets(folder_path, batch_size, max_len)
    tmp_x, tmp_y = train_set[0]
    x_shape = tmp_x.shape

    # set model
    model = build_model(x_shape,max_len)
    
    # set checkpointer
    checkpointer = ModelCheckpoint(filepath=save_path+'weights.hdf5', verbose=1, save_best_only=True)
    
    # set weight
    nsig = 2
    nbkg = 3
    w = np.concatenate([np.ones(nsig), np.zeros(nbkg)])
    class_weight = compute_class_weight('balanced', [0, 1], w) 
    
    # training
    history = model.fit_generator(
        generator = train_set,
        validation_data = val_set,
        steps_per_epoch = len(train_set), 
        epochs = epochs,
        callbacks = [checkpointer],
        #class_weight = class_weight
    )
    
    # save model image
    keras.utils.plot_model(model, to_file=save_path+'model_plot.png', show_shapes=True, show_layer_names=True)
    
    # evaluation
    train_s_res, train_b_res, test_s_res, test_b_res, test_y_true, test_y_score = evaluate(model, train_set, test_set)
    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']
   
    # save result informations
    np.savez(
        save_path+'model_info.npz',
        # learning curve
        y_vloss=y_vloss,
        y_loss = y_loss,
        train_len=len(train_set),
        val_len  =len(val_set),
        test_len =len(test_set),
        # ROC curve
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

