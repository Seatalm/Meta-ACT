#!/bin/bash

# Usage:
#   bash policy/ACT/eval_decomposed_hold_drawer_test.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [test_num] [extra overrides...]
#
# Example:
#   bash policy/ACT/eval_decomposed_hold_drawer_test.sh place_can_basket_decomposition demo_random_light demo_random_light 50 0 0 100
#   bash policy/ACT/eval_decomposed_hold_drawer_test.sh blocks_ranking_rgb_decomposition demo_clean_decomposition demo_clean_decomposition 50 0 3 100
#   bash policy/ACT/eval_decomposed_hold_drawer_test.sh put_object_cabinet_decomposition demo_clean_decomposition demo_clean_decomposition 50 0 2 100 \
#       --subtask1_expert_data_num 50 --subtask2_expert_data_num 150
#   bash policy/ACT/eval_decomposed_hold_drawer_test.sh blocks_ranking_rgb_decomposition demo_clean_decomposition demo_clean_decomposition 50 0 2 100 \
#       --subtask1_expert_data_num 50 --subtask2_expert_data_num 150 --subtask3_expert_data_num 100
policy_name=ACT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
test_num=${7:-100}
shift $(( $# >= 7 ? 7 : $# ))

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_decomposed_policy_hold_drawer_test.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_root policy/ACT/act_ckpt \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --temporal_agg true \
    --test_num ${test_num} \
    "$@"
