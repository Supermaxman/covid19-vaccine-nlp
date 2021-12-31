
import argparse
from collections import defaultdict
import os
from multiprocessing import Pool

import ujson as json
from tqdm import tqdm
import numpy as np
import pandas as pd


vec_size = 0
frame_map = {}


def embed_user(args):
	user_id, tweet_scores = args
	u_vec = np.zeros(shape=[vec_size], dtype=np.float32)
	u_vec_count = np.zeros(shape=[vec_size], dtype=np.float32)
	for tweet_id, frame_scores in tweet_scores.items():
		for frame_id, frame_score in frame_scores.items():
			for vec_idx in frame_map[frame_id]:
				u_vec[vec_idx] += frame_score
				u_vec_count[vec_idx] += 1.0
	# divide each vec_idx by the number of stances the user has on it
	# TODO make sparse
	u_vec /= np.clip(u_vec_count, a_min=1.0)
	return user_id, u_vec


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-f', '--frame_map_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-p', '--num_processes', default=12, type=int)
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

	# TODO build frame map
	with open(frame_map_path) as f:
		# [frame_id] -> List[vec_idx]
		frame_map = json.load(f)
	vec_size = max([max(v) for k, v in frame_map.items()]) + 1
	user_count = len(user_scores)
	with Pool(processes=num_processes) as p:
		for user_id, u_vec in tqdm(p.imap_unordered(embed_user, user_scores.items()), total=user_count):
			# TODO do something with user vectors
			# TODO either save or perform clustering here
			# TODO if saved then need to figure out save format. If I use a sparse matrix then need to keep
			# TODO user_id -> user_idx map
			pass


if __name__ == '__main__':
	main()
