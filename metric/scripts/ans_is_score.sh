#!/bin/bash

set -ex

for num in $(seq 0 4)
do
    echo "Number:" $num
    # nohup ./webui.sh 2>&1 | tee ../webui.log &
    python3 -u gpt2_is_score.py $num 2>&1 | tee "./log/is_score"$num".log" &
    sleep 1
done


