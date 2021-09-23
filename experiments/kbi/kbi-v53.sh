#!/usr/bin/env bash

filename=$(basename -- "$0")
run_id=${filename::-3}

seed=0
num_gpus=1
config=experiments/kbi/${run_id}.yaml

data_path=/users/max/code/covid19-vaccine-twitter/data/v1
train_path=${data_path}/train.jsonl
train_misinfo_path=${data_path}/misinfo.json
val_path=${data_path}/dev.jsonl
val_misinfo_path=${data_path}/misinfo.json
test_path=${data_path}/test.jsonl
test_misinfo_path=${data_path}/misinfo.json


export TOKENIZERS_PARALLELISM=true


echo "Starting experiment ${run_id}"
echo "Reserving ${num_gpus} GPU(s)..."

gpus=`request-gpus -r ${num_gpus}`
if [[ ${gpus} -eq -1 ]]; then
    echo "Unable to reserve ${num_gpus} GPU(s), exiting."
    exit -1
fi
echo "Reserved ${num_gpus} GPUs: ${gpus}"

# trap ctrl+c to free GPUs
handler()
{
    echo "Experiment ${run_id} aborted."
    echo "Freeing ${num_gpus} GPUs: ${gpus}"
    free-gpus -i ${gpus}
    exit -1
}
trap handler SIGINT

echo "Training ${run_id} model..."
python kbi/train.py \
  --config ${config} \
  --seed_everything ${seed} \
  --trainer.gpus ${gpus} \
  --trainer.default_root_dir models/${run_id} \
  --data.init_args.train_path ${train_path} \
  --data.init_args.train_misinfo_path ${train_misinfo_path} \
  --data.init_args.val_path ${val_path} \
  --data.init_args.val_misinfo_path ${val_misinfo_path}


echo "Freeing ${num_gpus} GPUs: ${gpus}"
free-gpus -i ${gpus}

echo "Experiment ${run_id} completed."