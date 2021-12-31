
import argparse
import pickle
from collections import defaultdict

import ujson as json
import sklearn.cluster as skc


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
