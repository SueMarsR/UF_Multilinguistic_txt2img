#!/bin/bash

set -ex
model_type="gpt2"
# model_type="flan-t5-base"

for idx in $(seq 0 0)
do
    rm -rf "data_comparison_model_output_pic/"$model_type"_"$idx
    mkdir -p "data_comparison_model_output_pic/"$model_type"_"$idx
done 

for num in $(seq 0 7)
do
    echo "Number:" $num
    # nohup ./webui.sh 2>&1 | tee ../webui.log &
    python3 -u 主实验_对比实验_ans_inference_multi_gpu.py $num $model_type 2>&1 | tee "./log/data_comparison_model_output_pic_"$model_type"_"$num".log" &
    sleep 1
done


