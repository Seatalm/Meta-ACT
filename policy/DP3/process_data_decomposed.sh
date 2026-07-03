#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
subtask_id=${4}

python scripts/process_data_decomposed.py ${task_name} ${task_config} ${expert_data_num} --subtask_id ${subtask_id}
