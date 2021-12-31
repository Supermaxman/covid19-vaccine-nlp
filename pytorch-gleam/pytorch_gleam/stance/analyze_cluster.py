
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
	parser.add_argument('-u', '--user_path', required=True)
	parser.add_argument('-th', '--threshold', default=0.05, type=float)
	args = parser.parse_args()

	input_path = args.input_path
	theme_path = args.theme_path
	user_path = args.user_path
	threshold = args.threshold

	with open(theme_path) as f:
		# ['idx'][theme_id] -> idx
		# ['text'][theme_id] -> text
		themes = json.load(f)
		t2idx = themes['idx']
		t2text = themes['text']
		idx2t = {v: k for k, v in t2idx.items()}
		idx2txt = {k: t2text[v] for k, v in idx2t.items()}

	print('collecting user vectors...')
	with open(user_path, 'rb') as f:
		profiles = pickle.load(f)
		user_ids = profiles['users']
		user_vecs = profiles['matrix']

	print(user_vecs.shape)
	print('loading clusters...')
	with open(input_path) as f:
		# [cluster_id]['users'] -> list[user_ids]
		# [cluster_id]['centroid'] -> list[float]
		clusters = json.load(f)

	total_users = user_vecs.shape[0]
	for cluster_id, cluster in sorted(clusters.items(), key=lambda x: len(x[1]['users'])):
		c_users = cluster['users']
		c_centroid = cluster['centroid']
		print(f'{cluster_id}: {len(c_users):,} users; {100*len(c_users)/total_users:.0f}%')
		for t_idx, t_text in idx2txt.items():
			t_score = c_centroid[t_idx]
			tax_name, tax_theme = idx2t[t_idx][1:-1].split(',')
			if abs(t_score) >= threshold:
				print(f'  {t_score:+.2f}: {t_text}')


if __name__ == '__main__':
	main()
