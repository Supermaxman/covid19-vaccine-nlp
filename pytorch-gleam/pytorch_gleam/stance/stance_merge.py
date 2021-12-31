
import argparse
from collections import defaultdict
import os
import ujson as json
from tqdm import tqdm
import numpy as np


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
	parser.add_argument('-f', '--tweet_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-t', '--threshold', default=0.28, type=float)
	args = parser.parse_args()

	input_path = args.input_path
	output_path = args.output_path
	tweet_path = args.tweet_path
	threshold = args.threshold
	user_lookup = {}
	print('building tweet-user index')
	for tweet in tqdm(read_jsonl(tweet_path), total=8161354):
		user_lookup[tweet['id']] = tweet['author_id']

	with open(input_path) as f:
		scores = json.load(f)

	print('collecting tweet-frame scores')
	tweets_kept = set()
	users_kept = set()
	users_seen = set()
	frame_count = defaultdict(int)
	keep_scores = defaultdict(lambda: defaultdict(dict))
	tweet_count = len(scores)
	for tweet_id, f_scores in tqdm(scores.items()):
		user_id = user_lookup[tweet_id]
		users_seen.add(user_id)
		for f_id, fs_scores in f_scores.items():
			not_rel_score, accept_score, reject_score = fs_scores
			if accept_score > threshold or reject_score > threshold:
				s_score = accept_score
				if accept_score < reject_score:
					s_score = -reject_score
				keep_scores[user_id][tweet_id][f_id] = s_score
				tweets_kept.add(tweet_id)
				users_kept.add(user_id)
				frame_count[f_id] += 1

	user_counts = np.array([float(len(v)) for k, v in keep_scores.items()])
	med_uc = np.median(user_counts)
	mean_uc = np.mean(user_counts)
	percentile_95_uc = np.percentile(user_counts, 95)
	percentile_80_uc = np.percentile(user_counts, 80)
	percentile_75_uc = np.percentile(user_counts, 75)
	min_uc = np.min(user_counts)
	max_uc = np.max(user_counts)
	f_counts = np.array([float(v) for k, v in frame_count.items()])
	med_fc = np.median(f_counts)
	mean_fc = np.mean(f_counts)
	min_fc = np.min(f_counts)
	min_f = [k for k, v in frame_count.items() if v == int(min_fc)][0]
	max_fc = np.max(f_counts)
	max_f = [k for k, v in frame_count.items() if v == int(max_fc)][0]
	print()
	print(f'{len(users_kept):,}/{len(users_seen):,} ({100 * len(users_kept)/len(users_seen):.0f}%) users')
	print(f'{len(tweets_kept):,}/{tweet_count:,} ({100 * len(tweets_kept)/tweet_count:.0f}%) tweets')
	print()
	print(f'{med_uc:,.2f} tweets / user (median)')
	print(f'{mean_uc:,.2f} tweets / user (mean)')
	print(f'{percentile_95_uc:,.2f} tweets / user (95 percentile)')
	print(f'{percentile_80_uc:,.2f} tweets / user (80 percentile)')
	print(f'{percentile_75_uc:,.2f} tweets / user (75 percentile)')
	print(f'{min_uc:,.0f} tweets / user (min)')
	print(f'{max_uc:,.0f} tweets / user (max)')
	print()
	print(f'{med_fc:,.2f} tweets / frame (median)')
	print(f'{mean_fc:,.2f} tweets / frame (mean)')
	print(f'{min_f}={min_fc:,.0f} tweets / frame (min)')
	print(f'{max_f}={max_fc:,.0f} tweets / frame (max)')

	print('saving scores')
	with open(output_path, 'w') as f:
		json.dump(keep_scores, f)


if __name__ == '__main__':
	main()
