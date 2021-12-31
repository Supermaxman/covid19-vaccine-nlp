
import argparse
import pickle
from collections import defaultdict

import numpy as np
import ujson as json
import sklearn.cluster as skc
import scipy.sparse as scp


def sim(a, b):
	a_b = -np.linalg.norm(a - b, axis=-1)
	return a_b


def cluster_kmeans(user_ids, user_vecs, num_clusters):
	model = skc.KMeans(
		n_clusters=num_clusters,
		random_state=0,
		n_init=20,
		verbose=0
	)

	model = model.fit(user_vecs)
	centroids = model.cluster_centers_
	cluster_users = defaultdict(list)
	clusters = {}
	for user_id, cluster_id in zip(user_ids, model.labels_):
		cluster_users[cluster_id].append(user_id)

	user_lookup = {user_id: idx for (idx, user_id) in enumerate(user_ids)}
	for cluster_id, cluster_users in cluster_users.items():
		cluster_centroid = centroids[cluster_id]

		c_user_vecs = scp.vstack([user_vecs[user_lookup[user_id]] for user_id in cluster_users]).toarray()
		user_sim = sim(c_user_vecs, np.expand_dims(cluster_centroid, axis=0))
		avg_sim = np.mean(user_sim)

		clusters[cluster_id] = {
			'users': cluster_users,
			'centroid': cluster_centroid.tolist(),
			'avg_centroid_dist': -avg_sim
		}
	return clusters


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-c', '--num_clusters', default=5, type=int)
	args = parser.parse_args()

	input_path = args.input_path
	output_path = args.output_path
	num_clusters = args.num_clusters

	print('collecting user vectors...')
	with open(input_path, 'rb') as f:
		profiles = pickle.load(f)
		user_ids = profiles['users']
		user_vecs = profiles['matrix']

	print(user_vecs.shape)
	print('clustering...')
	clusters = cluster_kmeans(user_ids, user_vecs, num_clusters)

	for cluster_id, cluster in clusters.items():
		print(f'{cluster_id}: {len(cluster["users"])}')

	print('saving clusters...')
	with open(output_path, 'w') as f:
		json.dump(clusters, f)


if __name__ == '__main__':
	main()
