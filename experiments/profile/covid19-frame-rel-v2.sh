#!/usr/bin/env bash

# run names
FILE_ID=covid19-frame-rel-v2

bm25_top_k=100000

# minimum number of tweets for each frame
min_rank=5000
# minimum relevance score for each frame
min_score=2.0

rel_gpus=0
rerank_model=nboost/pt-biobert-base-msmarco

artifacts_path=/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/artifacts
data_version=v4
data_root=/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.11.0.9-0.fc32.x86_64/

index_path=${data_root}/${data_version}/jsonl/tweets-index
small_index_path=${data_root}/${data_version}/jsonl/tweets-small-index-v2
question_path=${data_root}/${data_version}/jsonl-non-rt/covid_frame_questions.json

mkdir -p ${artifacts_path}
mkdir -p ${artifacts_path}/${FILE_ID}

output_path=${artifacts_path}/${FILE_ID}/${FILE_ID}
mkdir -p ${data_root}/${data_version}/jsonl/tweets-index-data-v2/


#	data_dir = 'data/covid19-retweets-raw'
#	jsonl_dir = '/users/max/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt'
gleam-parse-raw-tweet \


# rel/convert_tweets_to_jsonl
gleam-tweet-to-jsonl \
  --input_path ${data_root}/${data_version}/jsonl-non-rt/tweets.jsonl \
  --output_path ${data_root}/${data_version}/jsonl/tweets-index-data-v2/tweets.jsonl

gleam-tweet-to-jsonl \
  --input_path ${data_root}/${data_version}/jsonl-non-rt/retweets-v2.jsonl \
  --output_path ${data_root}/${data_version}/jsonl/tweets-index-data-v2/retweets.jsonl


python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -input ${data_root}/${data_version}/jsonl/tweets-index-data-v2 \
  -index ${small_index_path} \
  -storeDocvectors

# python rel/search_index_batch.py
gleam-search-tweet-index \
  --index_path ${small_index_path} \
  --query_path ${question_path} \
  --output_path ${output_path}_bm25_scores.json \
  --top_k ${bm25_top_k} \
  --threads 8

# rel/rerank_files.py
gleam-search-rerank \
  --index_path ${data_root}/${data_version}/jsonl/tweets-index-data \
  --questions_path ${question_path} \
  --scores_path ${output_path}_bm25_scores.json \
  # TODO get directory and format_rerank on dir
  --output_path ${output_path}_rerank_scores.json \
  --pre_model_name ${rerank_model} \
  --batch_size 64 \
  --max_seq_len 256 \
  --gpus ${rel_gpus}

python rel/format_rerank.py \
  --input_path ${output_path}_rerank_scores \
  --output_path ${output_path}_rerank_scores.json


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


#python rel/format_rerank.py \
#  --input_path ${output_path}_rerank_scores \
#  --output_path ${output_path}_rerank_scores.json

#python rel/select_candidates.py \
# --index_path ${index_path} \
# --questions_path ${question_path} \
# --scores_path ${output_path}_rerank_scores.json \
# --output_path ${output_path}_candidates.jsonl \
# --min_rank ${min_rank} \
# --min_score ${min_score}

python rel/select_candidates_files.py \
  --index_path ${data_root}/${data_version}/jsonl/tweets-index-data \
  --questions_path ${question_path} \
  --scores_path ${output_path}_rerank_scores.json \
  --output_path ${output_path}_candidates.jsonl \
  --min_rank ${min_rank} \
  --min_score ${min_score}