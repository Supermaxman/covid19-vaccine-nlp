
import argparse
import pickle
from multiprocessing import Pool

import ujson as json
from tqdm import tqdm
import numpy as np
import scipy.sparse as scp


vec_size = 0
frame_map = {}
mode = None


def embed_user(args):
	user_id, tweet_scores = args
	u_vec = np.zeros(shape=[vec_size], dtype=np.float32)
	u_vec_count = np.zeros(shape=[vec_size], dtype=np.float32)
	for tweet_id, frame_scores in tweet_scores.items():
		for frame_id, frame_score in frame_scores.items():
			# TODO some frames have no assigned taxonomy
			if frame_id not in frame_map:
				continue
			for vec_idx, vec_sign in frame_map[frame_id]:
				if mode == 'prob':
					u_vec[vec_idx] += vec_sign * frame_score
				elif mode == 'sign' or mode == 'cosine':
					u_vec[vec_idx] += vec_sign * np.sign(frame_score)
				else:
					raise ValueError(f'Unknown mode: {mode}')
				u_vec_count[vec_idx] += 1.0
	if u_vec_count.sum() == 0:
		return None, None
	# divide each vec_idx by the number of stances the user has on it
	u_vec /= np.maximum(u_vec_count, 1.0)
	if mode == 'cosine':
		u_vec /= np.linalg.norm(u_vec, axis=-1)
	u_vec = scp.csr_matrix(u_vec)
	return user_id, u_vec


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-f', '--frame_map_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-p', '--num_processes', default=12, type=int)
	parser.add_argument('-m', '--mode', default='prob')
	args = parser.parse_args()

	input_path = args.input_path
	frame_map_path = args.frame_map_path
	output_path = args.output_path
	num_processes = args.num_processes

	with open(input_path) as f:
		# [user_id][tweet_id][frame_id] -> frame_score
		user_scores = json.load(f)

	global frame_map
	global vec_size
	global mode

	mode = args.mode.lower()
	with open(frame_map_path) as f:
		# [frame_id] -> List[(vec_idx, score_sign)]
		frame_map = json.load(f)
	vec_size = max([max(v[0]) for k, v in frame_map.items()]) + 1
	user_count = len(user_scores)
	user_vecs = {
		'users': [],
		'matrix': []
	}
	with Pool(processes=num_processes) as p:
		for user_id, u_vec in tqdm(p.imap_unordered(embed_user, user_scores.items()), total=user_count):
			if user_id is not None:
				user_vecs['users'].append(user_id)
				user_vecs['matrix'].append(u_vec)

	user_vecs['matrix'] = scp.vstack(user_vecs['matrix'])
	with open(output_path, 'wb') as f:
		pickle.dump(user_vecs, f)


if __name__ == '__main__':
	main()
