import argparse
import pickle
from collections import defaultdict

import numpy as np
import ujson as json
from tqdm import tqdm


def dist(a, b):
	a_b = np.linalg.norm(a - b, axis=-1)
	return a_b


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
	parser.add_argument('-u', '--user_path', required=True)
	parser.add_argument('-t', '--tweets_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-c', '--num_samples', default=50, type=int)
	args = parser.parse_args()

	input_path = args.input_path
	user_path = args.user_path
	tweets_path = args.tweets_path
	output_path = args.output_path
	num_samples = args.num_samples

	print('collecting user vectors...')
	with open(user_path, 'rb') as f:
		profiles = pickle.load(f)
		user_ids = profiles['users']
		user_vecs = profiles['matrix']
		user_lookup = {user_id: idx for (idx, user_id) in enumerate(user_ids)}

	print('loading clusters...')
	with open(input_path, 'rb') as f:
		# [cluster_id]['users'] -> list[user_ids]
		# [cluster_id]['centroid'] -> list[float]
		clusters = pickle.load(f)

	print('collecting users for each cluster')
	cluster_samples = defaultdict(list)
	keep_users = set()
	for cluster_id, cluster in sorted(clusters.items(), key=lambda x: len(x[1]['users']), reverse=True):
		c_users = cluster['users']
		c_centroid = np.array(cluster['centroid'], dtype=np.float32)
		cluster_user_idxs = [user_lookup[user_id] for user_id in c_users]
		c_user_vecs = user_vecs[cluster_user_idxs]
		user_dists = dist(c_centroid, c_user_vecs)
		ind = np.argpartition(user_dists, -num_samples)[-num_samples:]
		ind = ind[np.argsort(user_dists[ind])[::-1]]
		for user_index in ind:
			sample_user_id = c_users[user_index]
			cluster_samples[cluster_id].append(
				sample_user_id
			)
			keep_users.add(sample_user_id)

	print('collecting tweets for each user...')
	users = defaultdict(list)
	for tweet in tqdm(read_jsonl(tweets_path), total=8161354):
		user_id = tweet['author_id']
		if user_id in keep_users:
			users[user_id].append(tweet)

	cluster_users = defaultdict(list)
	for cluster_id, sample_users in cluster_samples.items():
		for user_id in sample_users:
			cluster_users[cluster_id].append(
				{
					'user_id': user_id,
					'tweets': users[user_id]
				}
			)
	print(f'saving users...')
	with open(output_path, 'wb') as f:
		pickle.dump(cluster_users, f)


if __name__ == '__main__':
	main()
