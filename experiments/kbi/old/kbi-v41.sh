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
test_path=${data_path}/test.jsonl
test_misinfo_path=${data_path}/misinfo.json


export TOKENIZERS_PARALLELISM=true


echo "Starting experiment ${run_id}"
echo "Reserving ${num_gpus} GPU(s)..."

gpus=`python pytorch-gleam/pytorch_gleam/gpu/request_gpus.py -r ${num_gpus}`
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
    python pytorch-gleam/pytorch_gleam/gpu/free_gpus.py -i ${gpus}
    exit -1
}
trap handler SIGINT

#echo "Training model..."
python kbi/train.py \
  --seed_everything 0 \
  --model.ke_model transms \
  --model.ke_emb_size 8 \
  --model.ke_hidden_size 32 \
  --model.ke_gamma 1.5 \
  --model.ke_loss_norm 1 \
  --model.learning_rate 1e-4 \
  --trainer.max_epochs 40 \
  --data.batch_size 4 \
  --trainer.accumulate_grad_batches 8 \
  --trainer.check_val_every_n_epoch 4 \
  --data.pos_samples 2 \
  --data.neg_samples 2 \
  --data.max_seq_len 96 \
  --model.pre_model_name ${pre_model_name} \
  --data.tokenizer_name ${pre_model_name} \
  --data.num_workers 8 \
  --trainer.gpus ${gpus}, \
  --trainer.deterministic true \
  --trainer.default_root_dir models/${run_id} \
  --data.train_path ${train_path} \
  --data.train_misinfo_path ${train_misinfo_path} \
  --data.val_path ${val_path} \
  --data.val_misinfo_path ${val_misinfo_path}

#python kbi/validate.py \
#  --seed_everything 0 \
#  --model.ke_model transms \
#  --model.ke_emb_size 8 \
#  --model.ke_hidden_size 32 \
#  --model.ke_gamma 1.0 \
#  --model.ke_loss_norm 1 \
#  --model.learning_rate 5e-4 \
#  --trainer.max_epochs 40 \
#  --data.batch_size 4 \
#  --trainer.accumulate_grad_batches 8 \
#  --data.pos_samples 1 \
#  --data.neg_samples 1 \
#  --data.max_seq_len 96 \
#  --model.pre_model_name ${pre_model_name} \
#  --data.tokenizer_name ${pre_model_name} \
#  --data.num_workers 8 \
#  --trainer.gpus ${gpus}, \
#  --trainer.deterministic true \
#  --trainer.default_root_dir models/${run_id} \
#  --data.train_path ${train_path} \
#  --data.train_misinfo_path ${train_misinfo_path} \
#  --data.val_path ${val_path} \
#  --data.val_misinfo_path ${val_misinfo_path}


#python kbi/test.py \
#  --seed_everything 0 \
#  --model.ke_model transms \
#  --model.ke_emb_size 8 \
#  --model.ke_hidden_size 32 \
#  --model.ke_gamma 1.0 \
#  --model.ke_loss_norm 1 \
#  --model.learning_rate 5e-4 \
#  --trainer.max_epochs 40 \
#  --data.batch_size 4 \
#  --trainer.accumulate_grad_batches 8 \
#  --data.pos_samples 1 \
#  --data.neg_samples 1 \
#  --data.max_seq_len 96 \
#  --model.pre_model_name ${pre_model_name} \
#  --data.tokenizer_name ${pre_model_name} \
#  --data.num_workers 8 \
#  --trainer.gpus ${gpus}, \
#  --trainer.deterministic true \
#  --trainer.default_root_dir models/${run_id} \
#  --data.test_path ${test_path} \
#  --data.test_misinfo_path ${test_misinfo_path} \
#  --data.val_path ${val_path} \
#  --data.val_misinfo_path ${val_misinfo_path}


#echo "Testing model..."
# TODO PL_CONFIG
#python multi_class/test.py \
#  --seed_everything 0 \
#  --data.batch_size 16 \
#  --data.max_seq_len 128 \
#  --model.pre_model_name ${pre_model_name} \
#  --model.load_pre_model false \
#  --data.tokenizer_name ${pre_model_name} \
#  --data.num_workers 8 \
#  --trainer.gpus ${gpus}, \
#  --trainer.deterministic true \
#  --trainer.default_root_dir models/${run_id} \
#  --data.test_path ${test_path} \
#  --data.test_misinfo_path ${test_misinfo_path}

echo "Freeing ${num_gpus} GPUs: ${gpus}"
python pytorch-gleam/pytorch_gleam/gpu/free_gpus.py -i ${gpus}
