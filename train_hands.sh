#!/bin/bash
cd $(dirname $0)
seqs=(0051_dinosaur 0025_three_count)
for seq in ${seqs[@]}; do
    echo -e "\033[34m$seq\033[0m"
    torchrun --nproc_per_node=4 train.py --cfg_file configs/aninerf_interhand.yaml exp_name "0-$seq" capture 0 seq $seq resume False gpus "0,1,2,3" task "test" eval_ep 50
    echo -e "\033[34m*****************************************\033[0m"
done
