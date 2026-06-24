#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}
export MUJOCO_GL=egl
export SAPIEN_RENDERER=egl
export MPLCONFIGDIR=/tmp/matplotlib

PYTHONWARNINGS=ignore::UserWarning \
/data/s2wxy/conda_envs/RoboTwin/bin/python script/collect_data.py $task_name $task_config
rm -rf data/${task_name}/${task_config}/.cache
