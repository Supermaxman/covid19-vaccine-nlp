#!/usr/bin/env bash

data_path=/nas1-nfs1/data/maw150130/covid19
base_name=covid19-frame-rel-v2_stance
profile_type=sign
cluster_version=v2

#python pytorch-gleam/pytorch_gleam/stance/stance_profile.py \
#  --input_path ${data_path}/${base_name}-scores.json \
#  --frame_map_path ${data_path}/frame_map.json \
#  --output_path ${data_path}/${base_name}-profiles-m${profile_type}.pk \
#  --mode sign \
#  --num_processes 12

#for cluster_count in {3..10}
#do
#   python pytorch-gleam/pytorch_gleam/stance/profile_cluster.py \
#     --input_path ${data_path}/${base_name}-profiles-m${profile_type}.pk \
#     --output_path ${data_path}/${base_name}-clusters-m${profile_type}-v${cluster_version}-k${cluster_count}.pk \
#     --num_clusters ${cluster_count}
#done


cluster_count=9
python pytorch-gleam/pytorch_gleam/stance/analyze_cluster.py \
  --input_path ${data_path}/${base_name}-clusters-m${profile_type}-v${cluster_version}-k${cluster_count}.pk  \
  --user_path ${data_path}/${base_name}-profiles-m${profile_type}.pk \
  --threshold 0.1 \
  --theme_path ${data_path}/theme_map.json


python pytorch-gleam/pytorch_gleam/stance/collect_cluster_users.py \
  --input_path ${data_path}/${base_name}-clusters-m${profile_type}-v${cluster_version}-k${cluster_count}.pk  \
  --user_path ${data_path}/${base_name}-profiles-m${profile_type}.pk \
  --tweets_path ${data_path}/covid19-frame-rel-v2_candidates.jsonl \
  --output_path ${data_path}/${base_name}-clusters-m${profile_type}-v${cluster_version}-k${cluster_count}-users.pk \
  --num_samples 50

