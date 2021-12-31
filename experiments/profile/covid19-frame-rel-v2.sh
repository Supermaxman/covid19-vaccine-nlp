#!/usr/bin/env bash

# run names
FILE_ID=covid19-frame-rel-v2

artifacts_path=/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/artifacts
data_version=v4
data_root=/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.11.0.9-0.fc32.x86_64/

index_data_path=${data_root}/${data_version}/jsonl/tweets-index-data-v2
small_index_path=${data_root}/${data_version}/jsonl/tweets-small-index-v2
frame_path=/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/frames-covid19-parsed.jsonl

mkdir -p ${artifacts_path}
mkdir -p ${artifacts_path}/${FILE_ID}

output_path=${artifacts_path}/${FILE_ID}/${FILE_ID}
mkdir -p ${index_data_path}

gleam-parse-raw-tweet \
  --input_path /users/max/code/frame-vaccine-twitter/data/covid19-retweets-raw \
  --output_path ${data_root}/${data_version}/jsonl-non-rt/retweets-v2.jsonl

#gleam-tweet-to-jsonl \
#  --input_path ${data_root}/${data_version}/jsonl-non-rt/tweets.jsonl \
#  --output_path ${index_data_path}/tweets.jsonl

gleam-tweet-to-jsonl \
  --input_path ${data_root}/${data_version}/jsonl-non-rt/retweets-v2.jsonl \
  --output_path ${index_data_path}/retweets.jsonl


python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -input ${index_data_path} \
  -index ${small_index_path} \
  -storeDocvectors

gleam-search-tweet-index \
  --index_path ${small_index_path} \
  --query_path ${frame_path} \
  --output_path ${output_path}_bm25_scores.json \
  --top_k 400000 \
  --threads 8

#gleam-rerank \
#  --index_path ${index_data_path} \
#  --questions_path ${frame_path} \
#  --scores_path ${output_path}_bm25_scores.json \
#  --output_path ${output_path}_rerank_scores.json \
#  --pre_model_name nboost/pt-biobert-base-msmarco \
#  --batch_size 64 \
#  --max_seq_len 128 \
#  --gpus 0

# 27096
#python pytorch-gleam/pytorch_gleam/search/rerank.py \
#  --index_path data/covid19 \
#  --questions_path data/frames-covid19-parsed.jsonl \
#  --scores_path data/covid19-frame-rel-v2_bm25_scores.json \
#  --output_path data/covid19-frame-rel-v2_rerank_scores_fixed_test \
#  --pre_model_name nboost/pt-biobert-base-msmarco \
#  --batch_size 64 \
#  --max_seq_len 128 \
#  --num_workers 1 \
#  --gpus 5,6

#python pytorch-gleam/pytorch_gleam/search/rerank_format.py \
#--input_path data/covid19-frame-rel-v2_rerank_scores_fixed_test \
#--output_path data/covid19-frame-rel-v2_cross_rerank_scores_fixed_test.json


python pytorch-gleam/pytorch_gleam/search/rerank.py \
  --index_path data/covid19 \
  --questions_path data/frames-covid19-parsed.jsonl \
  --scores_path data/covid19-frame-rel-v2_bm25_scores.json \
  --output_path data/covid19-frame-rel-v2_rerank_scores_fixed \
  --pre_model_name nboost/pt-biobert-base-msmarco \
  --batch_size 64 \
  --max_seq_len 128 \
  --num_workers 1 \
  --gpus 2,3,4,5,6,7

python pytorch-gleam/pytorch_gleam/search/rerank_format.py \
--input_path data/covid19-frame-rel-v2_rerank_scores_fixed \
--output_path data/covid19-frame-rel-v2_cross_rerank_scores_fixed.json

gleam-rerank-format \
  --input_path ${output_path}_rerank_scores \
  --output_path ${output_path}_rerank_scores.json

#python ex/predict.py \
#  --config experiments/rerank/rerank-v1.yaml \
#  --trainer.gpus ${rel_gpus} \
#  --trainer.default_root_dir models/${FILE_ID}-rerank \
#  --data.batch_size 16 \
#  --data.max_seq_len 128 \
#  --data.predict_path ${data_root}/${data_version}/jsonl/tweets-index-data \
#  --data.predict_path ${data_root}/${data_version}/jsonl/tweets-index-data \
#  --data.frame_path ${frame_path}



#scp \
# max@hltgpu04:/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter/v4/jsonl/tweets-index-data/tweets.jsonl \
# data/tweets.jsonl
#scp \
# max@hltgpu04:/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter/v4/jsonl/tweets-index-data/retweets.jsonl \
# data/retweets.jsonl
#scp \
# max@hltgpu04:/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/artifacts/covid19-frame-rel-v1/covid19-frame-rel-v1_bm25_scores.json \
# data/covid19-frame-rel-v1_bm25_scores.json
#scp \
# max@hltgpu04:/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt/covid_frame_questions.json \
# data/covid_frame_questions.json

#python rel/rerank_files.py \
# --index_path data/covid19 \
# --questions_path data/covid_frame_questions.json \
# --scores_path data/covid19-frame-rel-v1_bm25_scores.json \
# --output_path data/covid19-frame-rel-v1_rerank_scores \
# --pre_model_name nboost/pt-biobert-base-msmarco \
# --batch_size 64 \
# --max_seq_len 256 \
# --gpus 5
#

python pytorch-gleam/pytorch_gleam/search/select_candidates.py \
  --index_path data/covid19 \
  --scores_path data/covid19-frame-rel-v2_cross_rerank_scores_fixed.json \
  --output_path data/covid19-frame-rel-v2_candidates.jsonl \
  --min_rank 5000 \
  --min_score 2.0

# minimum number of tweets for each frame
# minimum relevance score for each frame

gleam-search-candidates \
  --index_path ${index_data_path} \
  --scores_path ${output_path}_rerank_scores.json \
  --output_path ${output_path}_candidates.jsonl \
  --min_rank 5000 \
  --min_score 2.0

#rsync -avz --progress local/path/some_file usr@server.com:"/some/path/"


t-parse \
  --input_path ${output_path}_candidates.jsonl \
  --output_path ${output_path}_candidates-tparsed.jsonl

e-parse \
  --input_path ${output_path}_candidates-tparsed.jsonl \
  --frame_path ${frame_path} \
  --label_name candidates \
  --output_path ${output_path}_candidates-parsed.jsonl


python pytorch-gleam/pytorch_gleam/parse/efpparse.py \
  --input_path data/covid19-frame-rel-v2_candidates.jsonl \
  --frame_path data/frames-covid19-parsed.jsonl \
  --output_path /nas1-nfs1/data/maw150130/covid19/covid19-frame-rel-v2_candidates-parsed.jsonl \
  --num_processes 20


python pytorch-gleam/pytorch_gleam/parse/efpparse.py \
  --input_path ${output_path}_candidates.jsonl \
  --frame_path ${frame_path} \
  --output_path ${output_path}_candidates-parsed.jsonl \
  --num_processes 8


rsync -avz --progress \
  max@hltgpu04:"/users/max/code/covid19-vaccine-nlp/models/mcfmgcn-v36" \
  /nas1-nfs1/data/maw150130/covid19/models/mcfmgcn-v36



#bash ex/predict.sh experiments/profile/mcfmgcn-v36.yaml

python ex/predict.py \
  --config experiments/profile/mcfmgcn-v36.yaml \
  --trainer.default_root_dir models/mcfmgcn-v36 \
  --trainer.strategy ddp \
  --trainer.gpus 2,3,4,5,6,7


python pytorch-gleam/pytorch_gleam/callbacks/prediction.py \
  --input_path /nas1-nfs1/data/maw150130/covid19/stance-predictions \
  --output_path /nas1-nfs1/data/maw150130/covid19/covid19-frame-rel-v2_stance-predictions.json



python pytorch-gleam/pytorch_gleam/stance/stance_merge.py \
  --input_path /nas1-nfs1/data/maw150130/covid19/covid19-frame-rel-v2_stance-predictions.json \
  --tweet_path /nas1-nfs1/data/maw150130/covid19/covid19-frame-rel-v2_candidates.jsonl \
  --output_path /nas1-nfs1/data/maw150130/covid19/covid19-frame-rel-v2_stance-scores.json

python pytorch-gleam/pytorch_gleam/stance/stance_taxonomy.py \
  --input_path /home/maw150130/covid19-vaccine-nlp/data/taxonomy \
  --frame_map_output_path /nas1-nfs1/data/maw150130/covid19/frame_map.json \
  --theme_output_path /nas1-nfs1/data/maw150130/covid19/theme_map.json

bash experiments/profile/cluster-v1-k5.sh

bash experiments/profile/cluster-v2-k5.sh


