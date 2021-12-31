
import argparse
from collections import defaultdict

import ujson as json
from tqdm import tqdm
import numpy as np
import scipy.sparse as scp
import sklearn.cluster as skc


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


def cluster_kmeans(user_ids, user_vecs, num_clusters):
	model = skc.KMeans(
		n_clusters=num_clusters,
		random_state=0,
		n_init=20,
		verbose=1
	)

	model = model.fit(user_vecs)
	centroids = model.cluster_centers_
	cluster_users = defaultdict(list)
	clusters = {}
	for user_id, cluster_id in zip(user_ids, model.labels_):
		cluster_users[cluster_id].append(user_id)

	for cluster_id, cluster_users in cluster_users.items():
		cluster_centroid = centroids[cluster_id]
		clusters[cluster_id] = {
			'users': cluster_users,
			'centroid': cluster_centroid.tolist()
		}
	return clusters


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-t', '--theme_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-c', '--num_clusters', default=5, type=int)
	args = parser.parse_args()

	input_path = args.input_path
	theme_path = args.theme_path
	output_path = args.output_path
	num_clusters = args.num_clusters

	with open(theme_path) as f:
		# [theme_id] -> List[(vec_idx, score_sign)]
		theme_map = json.load(f)['idx']
	vec_size = max([v for k, v in theme_map.items()]) + 1

	user_ids = []
	user_vecs = []
	print('collecting user vectors...')
	for user in tqdm(read_jsonl(input_path), total=1425378):
		user_id = user['user_id']
		u_sparse = user['user_vec']
		user_ids.append(user_id)
		u_vec = np.zeros(shape=[vec_size], dtype=np.float32)
		for t_idx, t_score in u_sparse.items():
			t_idx = int(t_idx)
			t_score = float(t_score)
			u_vec[t_idx] = t_score
		u_vec = scp.csr_matrix(u_vec)
		user_vecs.append(u_vec)

	user_vecs = scp.vstack(user_vecs)
	print('clustering...')
	clusters = cluster_kmeans(user_ids, user_vecs, num_clusters)

	for cluster_id, cluster in clusters.items():
		print(f'{cluster_id}: {len(cluster["users"])}')

	print('saving clusters...')
	with open(output_path, 'w') as f:
		json.dump(clusters, f)


if __name__ == '__main__':
	main()
