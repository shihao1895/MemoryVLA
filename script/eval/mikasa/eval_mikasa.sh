#!/bin/bash

ckpts=(
/PATH/TO/CHECKPOINTS/ckpt1.pt
/PATH/TO/CHECKPOINTS/ckpt2.pt
)

task=all
num_eval=100
action_chunking_window=4
action_scale=1.0
gpu_id=0

for ckpt in "${ckpts[@]}"; do
    echo ">>> Evaluating: $ckpt"
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    python evaluation/mikasa/eval_mikasa_with_deploy.py \
        --pretrained_checkpoint "$ckpt" \
        --task "$task" \
        --num_eval_episodes "$num_eval" \
        --action_chunking_window "$action_chunking_window" \
        --action_scale "$action_scale" \
        --use_bf16
    echo ">>> Finished: $ckpt"
done

echo "========== All Done =========="
