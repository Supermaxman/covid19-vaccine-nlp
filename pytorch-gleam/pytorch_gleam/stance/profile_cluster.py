
import argparse
import pickle
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


def dist(a, b):
	a_b = np.linalg.norm(a - b, axis=-1)
	return a_b


def cluster_kmeans(user_ids, user_vecs, num_clusters, method):
	if method == 'kmeans':
		model = KMeans(
			n_clusters=num_clusters,
			random_state=0,
			n_init=20,
			verbose=0
		)
	else:
		raise ValueError(f'Unknown clustering method: {method}')

	# l2 normalize user embeddings
	uc_vecs = user_vecs / np.linalg.norm(user_vecs, axis=-1, keepdims=True)
	model = model.fit(uc_vecs)
	# l2 normalize centroid embeddings
	centroids = model.cluster_centers_ / np.linalg.norm(model.cluster_centers_, axis=-1, keepdims=True)
	cluster_users = defaultdict(list)
	clusters = {}
	for user_id, cluster_id in zip(user_ids, model.labels_):
		cluster_users[cluster_id].append(user_id)

	user_lookup = {user_id: idx for (idx, user_id) in enumerate(user_ids)}
	for cluster_id, cluster_users in cluster_users.items():
		cluster_centroid = centroids[cluster_id]
		cluster_user_idxs = [user_lookup[user_id] for user_id in cluster_users]
		c_uc_vecs = uc_vecs[cluster_user_idxs]
		user_sims = np.dot(cluster_centroid, c_uc_vecs.T)
		avg_sim = np.mean(user_sims)

		clusters[cluster_id] = {
			'sim': float(avg_sim),
			'users': cluster_users,
			'centroid': cluster_centroid.tolist()
		}
	return clusters


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-c', '--num_clusters', default=5, type=int)
	parser.add_argument('-m', '--method', default='kmeans')
	args = parser.parse_args()

	input_path = args.input_path
	output_path = args.output_path
	num_clusters = args.num_clusters
	method = args.method.lower()

	print('collecting user vectors...')
	with open(input_path, 'rb') as f:
		profiles = pickle.load(f)
		user_ids = profiles['users']
		user_vecs = profiles['matrix']

	print(user_vecs.shape)
	print('clustering...')
	clusters = cluster_kmeans(user_ids, user_vecs, num_clusters, method)

	for cluster_id, cluster in clusters.items():
		print(f'{cluster_id}: {len(cluster["users"])}')

	print('saving clusters...')
	with open(output_path, 'wb') as f:
		pickle.dump(clusters, f)


if __name__ == '__main__':
	main()
