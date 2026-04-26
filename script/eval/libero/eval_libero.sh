#!/bin/bash
set -euo pipefail
export MKL_INTERFACE_LAYER=GNU
export MUJOCO_GL=${MUJOCO_GL:-osmesa}

# PAI air-gapped env: HF cache + LLaMA local mirror staged by scripts/setup_libero.sh
# (no-ops on non-airgapped machines — guarded with ${:-default})
WORKSPACE=${WORKSPACE:-$(pwd)}
export HF_HOME=${HF_HOME:-$WORKSPACE/cache/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export MEMVLA_LLAMA2_7B_LOCAL_PATH=${MEMVLA_LLAMA2_7B_LOCAL_PATH:-$WORKSPACE/cache/llama2-7b-local}

ckpt_list=(
PATH_TO_CKPT_1
PATH_TO_CKPT_2
)

# libero_spatial, libero_object, libero_goal, libero_10, libero_90
task_suite_name=libero_spatial
gpu_id=0
action_chunking_window=8

num_trials_per_task=50
spcial_task_id=None
run_id_note="ac${action_chunking_window}"
unnorm_key="${task_suite_name}_no_noops"

find_free_port() {
  local min=${1:-2000}
  local max=${2:-30000}
  local port
  local tries=1000  # max tries to find a free port

  for ((i=0; i<tries; i++)); do
    port=$(shuf -i"${min}"-"${max}" -n1)
    if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
      echo "${port}"
      return 0
    fi
  done

  echo "ERROR: not found free port in range ${min}-${max}" >&2
  return 1
}

if [ "$spcial_task_id" != "None" ]; then
  spcial_task_id_arg="--spcial_task_id ${spcial_task_id}"
else
  spcial_task_id_arg=""
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}

for ckpt_path in "${ckpt_list[@]}"; do
  echo ">>> process ckpt：${ckpt_path}"
  port=$(find_free_port)
  echo "    port：${port}"

  local_log_dir="$(dirname "$(dirname "$ckpt_path")")/eval_libero/$(basename "$ckpt_path")"

  echo "Developing ..."
  python deploy.py \
    --saved_model_path ${ckpt_path} \
    --unnorm_key ${unnorm_key} \
    --adaptive_ensemble_alpha 0.1 \
    --cfg_scale 1.5 \
    --port ${port} \
    --action_chunking \
    --action_chunking_window ${action_chunking_window} &

  DEPLOY_PID=$!

  # NOTE: adjust sleep time according to your loading time, larger is better
  echo "sleeping 30 min to wait for model loading…"
  sleep 1800

  echo "Evaluating ..."
  python evaluation/libero/eval_libero.py \
    --task_suite_name ${task_suite_name} \
    --num_trials_per_task ${num_trials_per_task} \
    --run_id_note ${run_id_note} \
    --local_log_dir ${local_log_dir} \
    --port ${port} \
    ${spcial_task_id_arg}

  echo "kill developed service PID ${DEPLOY_PID}"
  kill ${DEPLOY_PID}
  echo ">>> finish ${ckpt_path}"
  echo
done

echo "All done!"
