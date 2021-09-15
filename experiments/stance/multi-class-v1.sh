#!/usr/bin/env bash

filename=$(basename -- "$0")
run_id=${filename::-3}

pre_model_name=digitalepidemiologylab/covid-twitter-bert-v2
num_gpus=1

data_path=/users/max/code/covid19-vaccine-twitter/data/v1
train_path=${data_path}/train.jsonl
train_misinfo_path=${data_path}/misinfo.json
val_path=${data_path}/dev.jsonl
val_misinfo_path=${data_path}/misinfo.json


export TOKENIZERS_PARALLELISM=true


echo "Starting experiment ${run_id}"
echo "Reserving ${num_gpus} GPU(s)..."

gpus=`python gpu/request_gpus.py -r ${num_gpus}`
if [[ ${gpus} -eq -1 ]]; then
    echo "Unable to reserve ${num_gpus} GPU(s), exiting."
    exit -1
fi
echo "Reserved ${num_gpus} GPUs: ${gpus}"

# trap ctrl+c to free GPUs
handler()
{
    echo "Experiment aborted."
    echo "Freeing ${num_gpus} GPUs: ${gpus}"
    python gpu/free_gpus.py -i ${gpus}
    exit -1
}
trap handler SIGINT

echo "Training model..."

python kbi/train.py \
  --seed_everything 0 \
  --model.learning_rate 5e-5 \
  --trainer.max_epochs 10 \
  --model.epochs 10 \
  --data.batch_size 8 \
  --data.max_seq_len 128 \
  --model.pre_model_name ${pre_model_name} \
  --data.tokenizer_name ${pre_model_name} \
  --data.collator_type multi_sequence \
  --data.num_workers 8 \
  --trainer.gpus ${gpus} \
  --trainer.deterministic true \
  --trainer.default_root_dir models \
  --data.train_path ${train_path} \
  --data.train_misinfo_path ${train_misinfo_path} \
  --data.val_path ${val_path} \
  --data.val_misinfo_path ${val_misinfo_path}


echo "Freeing ${num_gpus} GPUs: ${gpus}"
python gpu/free_gpus.py -i ${gpus}
