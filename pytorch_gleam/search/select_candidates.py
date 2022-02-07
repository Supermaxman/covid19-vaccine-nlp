
import os
import json
from collections import defaultdict

import argparse

from tqdm import tqdm


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					ex = json.loads(line)
					yield ex
				except Exception as e:
					print(e)


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


def get_tweets(dir_path):
	for file_name in os.listdir(dir_path):
		file_path = os.path.join(dir_path, file_name)
		for ex in read_jsonl(file_path):
			yield ex


def collect_tweets(dir_path, tweet_candidates):
	for tweet in tqdm(get_tweets(dir_path), total=19000000):
		if 'tweet' in tweet:
			tweet = tweet['tweet']
		tweet_id = tweet['id']
		if tweet_id not in tweet_candidates:
			continue
		tweet['candidates'] = tweet_candidates[tweet_id]
		yield tweet


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--index_path', required=True)
	parser.add_argument('-sc', '--scores_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-mir', '--min_rank', default=1, type=int)
	parser.add_argument('-mis', '--min_score', default=0.0, type=float)
	args = parser.parse_args()

	with open(args.scores_path) as f:
		# [q_id][t_id] = score
		scores = json.load(f)

	question_scores = defaultdict(list)
	for tweet_id, t_scores in scores.items():
		for q_id, score in t_scores.items():
			question_scores[q_id].append((score, tweet_id))

	print(f'Sorting tweets for each subquestion...')
	for q_id in tqdm(list(question_scores)):
		question_scores[q_id] = sorted(
			question_scores[q_id],
			# (score, tweet_id)
			key=lambda x: x[0],
			reverse=True
		)

	print(f'Collecting candidates for each tweet...')
	tweet_candidates = {}
	for q_id, q_rel in tqdm(question_scores.items()):
		for rank, (t_score, tweet_id) in enumerate(q_rel, start=1):
			if t_score < args.min_score and rank > args.min_rank:
				break
			if tweet_id not in tweet_candidates:
				# tweet['candidates'] = {}
				tweet_candidates[tweet_id] = {}
			t_candidates = tweet_candidates[tweet_id]
			t_candidates[q_id] = {
				'rank': rank,
				'score': t_score
			}

	print(f'Writing candidate tweets...')
	write_jsonl(
		collect_tweets(args.index_path, tweet_candidates),
		args.output_path
	)


if __name__ == '__main__':
	main()

