#!/usr/bin/env bash

config="$@"
num_gpus=1

config_name=`basename ${config}`
run_id=${config_name::-5}


export TOKENIZERS_PARALLELISM=true

echo "Starting experiment ${run_id}"
echo "Reserving ${num_gpus} GPU(s)..."

gpus=`request-gpus -r ${num_gpus}`
if [[ ${gpus} == "-1" ]]; then
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
python nli_text/train.py \
  --config ${config} \
  --trainer.gpus ${gpus} \
  --trainer.default_root_dir models/${run_id}

echo "Testing ${run_id} model..."
python nli_text/test.py \
  --config ${config} \
  --trainer.gpus ${gpus} \
  --trainer.default_root_dir models/${run_id}


echo "Freeing ${num_gpus} GPUs: ${gpus}"
free-gpus -i ${gpus}

echo "Experiment ${run_id} completed."
