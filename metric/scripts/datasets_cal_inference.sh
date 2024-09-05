#!/bin/bash

set -ex
data_type="eval"

rm -rf "images_multi_gpu/"$data_type
mkdir -p "images_multi_gpu/"$data_type


for num in $(seq 0 7)
do
    echo Number: "$num"
    # nohup ./webui.sh 2>&1 | tee ../webui.log &
    python3 -u inference_multi_gpu.py $num $data_type 2>&1 | tee ./log/"$data_type"_"$num".log &
    sleep 1
    # python3 -u inference_multi_gpu.py $num $data_type
done

# for ((i=1;i<=10;i++))
# do  
#     echo $i
# done

