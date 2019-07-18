import os,sys
import ROOT
from ROOT import TLorentzVector
from array import array
import numpy as np
import glob
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Bidirectional, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import Sequence 
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from pprint import pprint

class CMesonDataset(Sequence):

    def __init__(self, path, batch_size, max_len):
        self.root_file = ROOT.TFile(path)
        self.tree = self.root_file.delphys
        self.num_entries = self.tree.GetEntries()
        self.batch_size = batch_size
        self.max_len = max_len
        
        # Input X variables
        self.x_names = [
            'track_pt', 
            'track_deta',
            'track_dphi',
            'track_dz',
            'track_errd0',
            #'track_xd',
            #'track_yd',
            #'track_zd',
            'track_charge',
        ]
        
    def __len__(self):
        return int(self.num_entries / float(self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        
        x = []
        y = []
        for entry in range(start, end):
            self.tree.GetEntry(entry)
            
            # Set x 
            x_array = [np.array(getattr(self.tree, each), dtype=np.float32) for each in self.x_names]
            x_array = np.stack(x_array, axis=-1)
           
            # One hot incoding for PID 
            x_costompId = np.array(self.tree.track_costompId)
            x_pId = []
            for i in range(len(x_costompId)):
                if x_costompId[i] ==1:
                    x_pId.append([1,0,0])
                elif x_costompId[i] ==2:
                    x_pId.append([0,1,0])
                elif x_costompId[i] ==3:
                    x_pId.append([0,0,1])
                else:
                    x_pId.append([0,0,0])
            x_pId = np.array(x_pId)

            # Add PID to x array 
            x_array = np.concatenate([x_array, x_pId], axis=1)
            
            # Sort by PT
            #order_pt = np.argsort(self.tree.track_pt)[::-1]
            #x_array = x_array[order_pt]
            x.append(x_array)
            
            # Set y 
            y_array = [np.array(self.tree.dau_label, dtype=np.int64)]
            y_array = np.stack(y_array, axis=-1)
            
            #y_array = y_array[order_pt]
            y.append(y_array)

        x = keras.preprocessing.sequence.pad_sequences(x, maxlen=self.max_len, padding='post', truncating='post', dtype=np.float32)
        y = keras.preprocessing.sequence.pad_sequences(y, maxlen=self.max_len, padding='post', truncating='post', dtype=np.int64)
        return x, y

def get_datasets(folder_path, batch_size, max_len):
    dataset = glob.glob(folder_path+'*.root')
    datasets = [
        CMesonDataset(dataset[0], batch_size, max_len),
        CMesonDataset(dataset[1], batch_size, max_len),
        CMesonDataset(dataset[2], batch_size, max_len),
    ]
    
    train_set, val_set, test_set = sorted(datasets, key=lambda dset: len(dset), reverse=True)
    return train_set, val_set, test_set

def main():
    batch_size = 10
    max_len = 30

    folder_name = sys.argv[1]	
    folder_path = '../3-Selector/{}/'.format(folder_name)	
    train_set, val_set, test_set = get_datasets(folder_path, batch_size, max_len)

    print"Train Set : ", len(train_set) 
    print"Val Set : ", len(val_set) 
    print"Test Set : ", len(test_set) 
    
    tmp_x, tmp_y = train_set[5]
    print tmp_x
    print tmp_y
    print tmp_x.shape, tmp_y.shape
    

if __name__ == '__main__':
    main()
