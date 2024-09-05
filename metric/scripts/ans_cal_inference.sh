#!/bin/bash

set -ex
model_type="gpt2"

for idx in $(seq 0 4)
do
    rm -rf "model_output_pic/"$model_type"_"$idx
    mkdir -p "model_output_pic/"$model_type"_"$idx
done 

for num in $(seq 0 7)
do
    echo "Number:" $num
    # nohup ./webui.sh 2>&1 | tee ../webui.log &
    python3 -u ans_inference_multi_gpu.py $num $model_type 2>&1 | tee "./log/model_output_pic_"$model_type"_"$num".log" &
    sleep 1
done


