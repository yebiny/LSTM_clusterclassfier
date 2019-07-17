import numpy as np
import os,sys
import matplotlib as mpl
mpl.use("Agg")
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

folder = './'+sys.argv[1:][0]

info = np.load(folder+"/info_eval.npz")
y_true = info['test_y_true' ]
y_score =info['test_y_score']

# true positive rates
fpr, tpr, _ = metrics.roc_curve(y_true , y_score[:, 0])
# true negative rates 
tnr = 1-fpr
auc = metrics.auc(x=tpr, y=tnr)

# plot
title = "ROC Curve"
label = "RNN (AUC = {:.3f})".format(auc)
fig, ax = plt.subplots(figsize=(8,6))
roc_curve = Line2D(
    xdata=tpr, ydata=tnr,
    label=label,
    color='darkorange', alpha=0.8, lw=3)

ax.add_line(roc_curve)
ax.set_xlabel('True Positive Rates (Signal Efficiency)',fontsize=12)
ax.set_ylabel('True Negative Rates (Background Rejection)', fontsize=12)
ax.grid()
ax.legend()
ax.set_title("ROC Curve",fontsize=15)

fig.savefig('./'+folder+'/roc_auc.png')
plt.close()
