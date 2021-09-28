#!/usr/bin/env bash

run_id=mc-covaxlies-v1

pre_model_name=digitalepidemiologylab/covid-twitter-bert-v2
num_gpus=1

train_path=data/train.jsonl
train_misinfo_path=data/misinfo.json
val_path=data/dev.jsonl
val_misinfo_path=data/misinfo.json
test_path=data/test.jsonl
test_misinfo_path=data/misinfo.json


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
    echo "Experiment aborted."
    echo "Freeing ${num_gpus} GPUs: ${gpus}"
    free-gpus -i ${gpus}
    exit -1
}
trap handler SIGINT

#echo "Training model..."
# PL_CONFIG=config_path to set from config
# TODO checkpoint callback removal, set to only save at end
python multi_class/train.py \
  --seed_everything 0 \
  --model.learning_rate 5e-5 \
  --trainer.max_epochs 10 \
  --data.batch_size 8 \
  --data.max_seq_len 128 \
  --model.pre_model_name ${pre_model_name} \
  --data.tokenizer_name ${pre_model_name} \
  --data.num_workers 8 \
  --trainer.gpus ${gpus} \
  --trainer.deterministic true \
  --trainer.default_root_dir models/${run_id} \
  --data.train_path ${train_path} \
  --data.train_misinfo_path ${train_misinfo_path} \
  --data.val_path ${val_path} \
  --data.val_misinfo_path ${val_misinfo_path}

echo "Testing model..."
python multi_class/test.py \
  --seed_everything 0 \
  --data.batch_size 16 \
  --data.max_seq_len 128 \
  --model.pre_model_name ${pre_model_name} \
  --model.load_pre_model false \
  --data.tokenizer_name ${pre_model_name} \
  --data.num_workers 8 \
  --trainer.gpus ${gpus} \
  --trainer.deterministic true \
  --trainer.default_root_dir models/${run_id} \
  --data.test_path ${test_path} \
  --data.test_misinfo_path ${test_misinfo_path}

echo "Freeing ${num_gpus} GPUs: ${gpus}"
free-gpus -i ${gpus}
