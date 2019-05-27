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

    def __init__(self, path, batch_size):
        self.root_file = ROOT.TFile(path)
        self.tree = self.root_file.delphys
        self.num_entries = self.tree.GetEntries()
        self.batch_size = batch_size
        
    def __len__(self):
        return int(self.num_entries / float(self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        
        x = []
        y = []

        for entry in range(start, end):
                       
            self.tree.GetEntry(entry)
            x_pt = np.array(list(self.tree.track_pt), dtype=np.float32)
            x_deta = np.array(list(self.tree.track_deta), dtype=np.float32)
            x_dphi = np.array(list(self.tree.track_dphi), dtype=np.float32)
            x_d0 = np.array(list(self.tree.track_d0), dtype=np.float32)
            x_dz = np.array(list(self.tree.track_dz), dtype=np.float32)
            x_xd = np.array(list(self.tree.track_xd), dtype=np.float32)
            x_yd = np.array(list(self.tree.track_yd), dtype=np.float32)
            x_zd = np.array(list(self.tree.track_zd), dtype=np.float32)
 
            #Sorting
            order_pt = np.argsort(x_pt)
                                 
            x_pt = x_pt[order_pt][::-1]
            x_deta = x_deta[order_pt][::-1]
            x_dphi = x_dphi[order_pt][::-1]
            x_d0 = x_d0[order_pt][::-1]
            x_dz = x_dz[order_pt][::-1]
            x_xd = x_xd[order_pt][::-1]  
            x_yd = x_yd[order_pt][::-1]    
            x_zd = x_zd[order_pt][::-1]
            
            x_set = []
                
            for i in range(0, len(x_pt)):
                    x_set.append([])
                    x_set[-1].append(x_pt[i])
                    x_set[-1].append(x_deta[i])
                    x_set[-1].append(x_dphi[i])
                    x_set[-1].append(x_d0[i])
                    x_set[-1].append(x_dz[i])
                    x_set[-1].append(x_xd[i])
                    x_set[-1].append(x_yd[i])
                    x_set[-1].append(x_zd[i])
         
            x.append(x_set)
            
            if (self.tree.jet_label ==3): y_set = 1
            else : y_set = 0
            
            y.append(y_set)

        x = keras.preprocessing.sequence.pad_sequences(x, maxlen=15, padding='post', truncating='post', dtype=np.float32)
        y = np.array(y)
        
        return x, y
    
def get_datasets(folder_path):
    dataset = glob.glob(folder_path+'*.root')
    print(dataset)	
    datasets = [
        CMesonDataset(dataset[0], batch_size=256),
        CMesonDataset(dataset[1], batch_size=256),
        CMesonDataset(dataset[2], batch_size=256),
    ]
    
    train_set, val_set, test_set = sorted(datasets, key=lambda dset: len(dset), reverse=True)
    return train_set, val_set, test_set

def main():
    folder_name = sys.argv[1]	
    folder_path = '../3-Selector/{}/'.format(folder_name)	
    print(folder_name, folder_path)
    train_set, val_set, test_set = get_datasets(folder_path)

    print("Train Set : ",train_set, len(train_set) )
    print("Val Set : ",val_set, len(val_set) )
    print("Test Set : ",test_set, len(test_set) )
    print(train_set[0])

if __name__ == '__main__':
    main()
