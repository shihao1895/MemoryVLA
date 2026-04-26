#!/bin/bash

# PAI air-gapped env: HF cache + LLaMA local mirror staged by scripts/setup_simpler.sh
# (these are no-ops if the machine isn't air-gapped — they're guarded with ${:-default})
WORKSPACE=${WORKSPACE:-$(pwd)}
export HF_HOME=${HF_HOME:-$WORKSPACE/cache/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export MEMVLA_LLAMA2_7B_LOCAL_PATH=${MEMVLA_LLAMA2_7B_LOCAL_PATH:-$WORKSPACE/cache/llama2-7b-local}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}

ckpt_paths=(
/PATH/TO/YOUR/CHECKPOINT_1
/PATH/TO/YOUR/CHECKPOINT_2
such as: ./checkpoints/step-032500-epoch-03-loss=0.0455.pt
...
)

gpu_id=0

for ckpt_path in "${ckpt_paths[@]}"; do
    eval_dir=$(dirname $(dirname ${ckpt_path}))/eval_simpler/$(basename ${ckpt_path})
    mkdir -p ${eval_dir}

    scene_name=bridge_table_1_v1
    robot=widowx
    rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
    robot_init_x=0.147
    robot_init_y=0.028

    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py --ckpt-path ${ckpt_path} \
      --robot ${robot} --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${eval_dir}/Cube.txt;

    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py --ckpt-path ${ckpt_path} \
      --robot ${robot} --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${eval_dir}/Carrot.txt;

    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py --ckpt-path ${ckpt_path} \
      --robot ${robot} --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${eval_dir}/Spoon.txt;

    scene_name=bridge_table_1_v2
    robot=widowx_sink_camera_setup
    rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
    robot_init_x=0.127
    robot_init_y=0.06

    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py --ckpt-path ${ckpt_path} \
      --robot ${robot} --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${eval_dir}/Eggplant.txt;

  wait
  echo "Done: ${ckpt_path}"
done

wait
echo "All done!"
