#!/usr/bin/env bash

filename=$(basename -- "$0")
run_id=${filename::-3}

pre_model_name=digitalepidemiologylab/covid-twitter-bert-v2
num_gpus=1
learning_rate=5e-5
ke_model=pytorch_gleam.modeling.knowledge_embedding.TransMSEmbedding
ke_emb_size=8
ke_hidden_size=32
ke_gamma=1.5
ke_loss_norm=1
max_epochs=48
batch_size=4
accumulate_grad_batches=8
pos_samples=1
neg_samples=1
check_val_every_n_epoch=4
max_seq_len=96
num_workers=8
seed=0

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

#echo "Training model..."
python kbi/train.py \
  --seed_everything ${seed} \
  --model.ke ${ke_model} \
  --model.ke.emb_size ${ke_emb_size} \
  --model.ke.hidden_size ${ke_hidden_size} \
  --model.ke.gamma ${ke_gamma} \
  --model.ke.loss_norm ${ke_loss_norm} \
  --model.threshold pytorch_gleam.modeling.thresholds.MultiClassCallableThresholdModule \
  --model.metric pytorch_gleam.modeling.metrics.F1PRMultiClassMetric \
  --model.metric.mode macro \
  --model.metric.num_classes 3 \
  --model.learning_rate ${learning_rate} \
  --trainer.max_epochs ${max_epochs} \
  --data.batch_size ${batch_size} \
  --trainer.accumulate_grad_batches ${accumulate_grad_batches} \
  --trainer.check_val_every_n_epoch ${check_val_every_n_epoch} \
  --data.pos_samples ${pos_samples} \
  --data.neg_samples ${neg_samples} \
  --data.max_seq_len ${max_seq_len} \
  --model.pre_model_name ${pre_model_name} \
  --model.load_pre_model true \
  --data.tokenizer_name ${pre_model_name} \
  --data.num_workers ${num_workers} \
  --trainer.gpus ${gpus} \
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


echo "Freeing ${num_gpus} GPUs: ${gpus}"
free-gpus -i ${gpus}

echo "Experiment ${run_id} completed."