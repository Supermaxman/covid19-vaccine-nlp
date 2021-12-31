#!/usr/bin/env bash

data_path=/nas1-nfs1/data/maw150130/covid19
base_name=covid19-frame-rel-v2_stance

python pytorch-gleam/pytorch_gleam/stance/stance_profile.py \
  --input_path ${data_path}/${base_name}-scores.json \
  --frame_map_path ${data_path}/frame_map.json \
  --output_path ${data_path}/${base_name}-profiles.pk \
  --num_processes 12

python pytorch-gleam/pytorch_gleam/stance/profile_cluster.py \
  --input_path ${data_path}/${base_name}-profiles.pk \
  --output_path ${data_path}/${base_name}-clusters-v1-k5.json \
  --num_clusters 5

python pytorch-gleam/pytorch_gleam/stance/analyze_cluster.py \
  --input_path ${data_path}/${base_name}-clusters-v1-k5.json  \
  --user_path ${data_path}/${base_name}-profiles.pk \
  --theme_path ${data_path}/theme_map.json




