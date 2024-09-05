#!/bin/bash

set -ex
# model_type="gpt2"
# model_type="flan-t5-base"
picture_path_list=(
    "./model_output_pic/gpt2_0"
    "./model_output_pic/gpt2_1"
    "./model_output_pic/gpt2_2"
    "./model_output_pic/gpt2_3"
    "./model_output_pic/gpt2_4"
    "./data_comparison_model_output_pic/gpt2_0"
    "./data_comparison_model_output_pic/flan-t5-base_0"
    "./images_multi_gpu/eval"
)

arraylength=${#picture_path_list[@]}

for (( i=0; i<${arraylength}; i++ ));
do
    # echo $i
    # echo ${picture_path_list[$i]}
    python3 -u inference_iqa_multi_gpu.py $i ${picture_path_list[$i]} &
    sleep 10
done
# for num in $(seq 0 7)
# do
#     echo "Number:" $num
#     # nohup ./webui.sh 2>&1 | tee ../webui.log &
#     python3 -u 主实验_对比实验_ans_inference_multi_gpu.py $num $model_type 2>&1 | tee "./log/data_comparison_model_output_pic_"$model_type"_"$num".log" &
#     sleep 1
# done


