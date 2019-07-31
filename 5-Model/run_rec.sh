#!/bin/sh
if [ "$(hostname)" == "gate2.sscc.uos.ac.kr" ]
then
    echo "Gate2"
    source /opt/ROOT/bin/thisroot.sh
elif [ "$(hostname)" == "dgx" ]
then
    echo "DGX"
    source /opt/root_v6.16.00.Linux-centos7-x86_64-gcc4.8/bin/thisroot.sh
else
    echo "Unknown host: ${HOSTNAME}"
    exit 1
fi

python /home/yyoun/deepcmeson/5-Model/model_rec.py $@
