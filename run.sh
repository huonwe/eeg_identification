#!/bin/bash
cur_dir=$(cd $(dirname $0)  && pwd)
parent_dir=$(cd $(dirname $0) && cd .. && pwd)


if [ $# -ne 4 ] ; then
	echo "***********************************************************************************"
	echo "请输入三个shell脚本参数：模型名称，通道区域名称，GPU序号"
	echo "模型名称：# cnn  / vcnn/ n_vcnn / lstm / gru 2/cnn_lstm / cnn_gru /cnn_blstm /n_vcnn_lstm  / n_vcnn_gru/ n_vcnn_blstm/vcnn_lstm / vcnn_gru  / vcnn_blstm/ "
	echo "通道区域名称：P/C/CP/P-C/C-CP/P-CP/P-C-CP/64/4/8/16/32"
	echo "GPU序号：0/1"
	echo "Epochs 数量:200/400/600/800/1000"
	echo "bash run.sh  model_name   eeg_channel gpu_number  epoch_number" 
	echo "例如：bash run.sh  cnn_pure   ALL  0  400"
	echo "***********************************************************************************"
	exit 1
fi
model_name=$1
eeg_channel=$2
gpu_number=$3
epoch_number=$4


if [ -d ./log/${model_name}/${epoch_number} ]; then
     echo "yes"
else
    echo "no"
    mkdir -p  ./log/${model_name}/${epoch_number}/
fi

nohup python -u train_torch.py ${model_name}  --channel ${eeg_channel}  --ctx ${gpu_number} --maxEpoch ${epoch_number}> ./log/${model_name}/${epoch_number}/train-${model_name}-${eeg_channel}.log 2>&1 &

ps  -aux |grep train_torch.py
nvidia-smi 


sleep 15
cat  ./log/${model_name}/${epoch_number}/train-${model_name}-${eeg_channel}.log

#tail -f  ./log/${model_name}/train-${model_name}-${eeg_channel}.log



