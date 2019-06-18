#!/bin/sh
project=${1}
echo "-------- Draw Loss, Acc ----------"
python loss.py ${project}
echo "-------- Draw Resoponce ----------"
python responce.py ${project}
echo "-------- Draw ROC curve ----------"
python roc.py ${project}

