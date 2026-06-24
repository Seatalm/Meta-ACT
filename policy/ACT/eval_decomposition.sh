#!/bin/bash

# == keep unchanged ==
policy_name=ACT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
subtask_id=${7:-0}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
if [ "$subtask_id" -gt 0 ]; then
    echo -e "\033[33mRunning Subtask: ${subtask_id}\033[0m"
fi

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_decomposition.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --subtask_id ${subtask_id} \
    --ckpt_dir policy/ACT/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num} \
    --seed ${seed} \
    --temporal_agg true