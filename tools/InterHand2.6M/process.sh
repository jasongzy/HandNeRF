#!/bin/bash
hand_type=right
split=test
capture=0
seqs=(ROM03_RT_No_Occlusion ROM04_RT_Occlusion)
for seq in ${seqs[@]}; do
    echo -e "\033[34m$seq\033[0m"
    python get_annots.py --hand_type $hand_type --split $split --capture $capture --seq $seq
    python prepare_blend_weights.py --hand_type $hand_type --split $split --capture $capture --seq $seq
    python render_mesh.py --hand_type $hand_type --split $split --capture $capture --seq $seq
    echo -e "\033[34m*****************************************\033[0m"
done
