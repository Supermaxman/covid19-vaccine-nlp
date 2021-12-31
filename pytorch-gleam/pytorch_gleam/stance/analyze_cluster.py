import argparse
import pickle

import numpy as np
import ujson as json
import scipy.sparse as scp


def sim(a, b):
	a_b = -np.linalg.norm(a - b, axis=-1)
	return a_b


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
		user_lookup = {user_id: idx for (idx, user_id) in enumerate(user_ids)}

	print(user_vecs.shape)
	print('loading clusters...')
	with open(input_path) as f:
		# [cluster_id]['users'] -> list[user_ids]
		# [cluster_id]['centroid'] -> list[float]
		clusters = json.load(f)

	total_users = user_vecs.shape[0]
	for cluster_id, cluster in sorted(clusters.items(), key=lambda x: len(x[1]['users']), reverse=True):
		c_users = cluster['users']
		c_centroid = cluster['centroid']
		c_avg_centroid_dist = cluster['avg_centroid_dist']

		print(
			f'Cluster {cluster_id}: {len(c_users):,} '
			f'users ({100 * len(c_users) / total_users:.0f}%) '
			f'{c_avg_centroid_dist:.2f} avg centroid distance'
		)
		current_tax = None
		for t_idx, t_text in idx2txt.items():
			t_score = c_centroid[t_idx]
			tax_name, tax_theme = idx2t[t_idx][1:-1].split(',')
			tax_name = tax_name[1:-1]
			if tax_name == 'literacy':
				if t_text == '+':
					t_text = 'Having literacy'
				else:
					t_text = 'Lacking literacy'
			elif tax_name == 'civil_rights':
				tax_name = 'Civil rights'
				if t_text == '+':
					t_text = 'Vaccines more important than civil rights'
				else:
					t_text = 'Civil rights above all'
			elif tax_name == 'trust+':
				tax_name = 'Building trust'
			elif tax_name == 'trust-':
				tax_name = 'Eroding trust'
			if abs(t_score) >= threshold:
				if current_tax != tax_name:
					current_tax = tax_name
					tax_name = tax_name.title()
					print(f'  {tax_name} Taxonomy')
				print(f'    {t_score:+.2f}: {t_text}')
		print('-------')


if __name__ == '__main__':
	main()
