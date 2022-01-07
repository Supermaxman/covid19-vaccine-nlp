import argparse
import pickle
from collections import defaultdict

import numpy as np
import ujson as json
from tqdm import tqdm


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-s', '--scores_path', required=True)
	parser.add_argument('-t', '--tweets_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	input_path = args.input_path
	scores_path = args.scores_path
	tweets_path = args.tweets_path
	output_path = args.output_path

	print('loading scores...')
	with open(scores_path, 'r') as f:
		scores = json.load(f)

	print('loading clusters...')
	with open(input_path, 'rb') as f:
		# [cluster_id]['users'] -> list[user_ids]
		# [cluster_id]['centroid'] -> list[float]
		clusters = pickle.load(f)

	print('collecting users for each cluster')
	user_cluster = {}
	for cluster_id, cluster in clusters.items():
		c_users = cluster['users']
		for user_id in c_users:
			user_cluster[user_id] = cluster_id

	cluster_frame_stances = defaultdict(lambda: defaultdict(dict))
	print('collecting tweets for each user...')
	for tweet in tqdm(read_jsonl(tweets_path), total=8161354):
		user_id = tweet['author_id']
		tweet_id = tweet['id']
		if user_id not in user_cluster:
			continue
		user_cluster_id = user_cluster[user_id]
		if user_id in scores:
			u_scores = scores[user_id]
			if tweet_id in u_scores:
				ut_scores = u_scores[tweet_id]
				for f_id, f_score in ut_scores.items():
					f_stance = 'Accept'
					if f_score < 0:
						f_stance = 'Reject'
					cluster_frame_stances[user_cluster_id][f_id][f_stance] += 1

	print(f'saving stats...')
	with open(output_path, 'w') as f:
		json.dump(cluster_frame_stances, f)


if __name__ == '__main__':
	main()
