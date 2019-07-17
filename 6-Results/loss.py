from __future__ import division
from collections import OrderedDict
import numpy as np
from ROOT import TH1F
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import os,sys

folder = './'+sys.argv[1:][0]
info = np.load(folder+"/info_learning.npz")
#info = np.load(folder+"/info_eval.npz")
#info = np.load(folder+"/CSVLogger.csv")
vloss=info['y_vloss']
loss =info['y_loss' ]
vacc=info['y_vacc']
acc =info['y_acc' ]

x_len = np.arange(len(loss))
plt.plot(x_len, vacc,marker='.', c = 'darkorange', label = 'Acc: Valdation-set')
plt.plot(x_len, acc, marker='.', c = 'green',label = 'Acc: Train-Set')
plt.plot(x_len, vloss,marker='.', c = 'red', label = 'Loss: Valdation-set')
plt.plot(x_len, loss, marker='.', c = 'blue',label = 'Loss: Train-Set')
plt.legend(fontsize=10)
plt.grid()
plt.savefig(folder+"/loss.png")
 
