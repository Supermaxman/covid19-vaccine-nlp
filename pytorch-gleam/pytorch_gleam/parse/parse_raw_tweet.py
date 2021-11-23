import argparse
import os
import json
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from multiprocessing import Pool


def read_file(file_path):
	with open(file_path) as f:
		data = json.load(f)
	return data


def invert_errors(errors):
	inv = defaultdict(dict)
	for error in errors:
		r_id = error['resource_id']
		e_type = error['detail'].replace(f': [{r_id}].', '')
		inv[r_id][e_type] = error
	return inv


def invert_ids(items):
	inv = {}
	for item in items:
		item_id = item['id'] if 'id' in item else item['media_key']
		inv[item_id] = item
		if 'username' in item:
			inv[item['username']] = item
	return inv


def invert_includes(includes):
	inv = {}
	for key, vals in includes.items():
		inv[key] = invert_ids(vals)
	return inv


def parse_tweet(tweet, inv_includes, inv_errors):
	author_id = tweet['author_id']
	if author_id in inv_errors:
		author = inv_errors[author_id]
	else:
		author = inv_includes['users'][author_id]
	tweet['author'] = author
	if 'entities' not in tweet:
		tweet['entities'] = {}
	for e_type, e_vals in tweet['entities'].items():
		if e_type == 'mentions':
			for e in e_vals:
				if 'username' in e:
					e_username = e['username']
					if e_username in inv_errors:
						e_user = inv_errors[e_username]
					elif e_username in inv_includes['users']:
						e_user = inv_includes['users'][e_username]
					else:
						e_user = None
					e['user'] = e_user
	if 'referenced_tweets' not in tweet:
		tweet['referenced_tweets'] = []
	for ref_tweet in tweet['referenced_tweets']:
		r_id = ref_tweet['id']
		if r_id in inv_errors:
			r_tweet = inv_errors[r_id]
		elif r_id in inv_includes['tweets']:
			r_tweet = inv_includes['tweets'][r_id]
		else:
			r_tweet = None
		ref_tweet['data'] = r_tweet
	return tweet


def parse_tweets(tweets):
	if 'data' not in tweets:
		return
	t_data = tweets['data']
	if 'includes' not in tweets:
		tweets['includes'] = {}
	if 'errors' not in tweets:
		tweets['errors'] = []
	t_includes = invert_includes(tweets['includes'])
	t_errors = invert_errors(tweets['errors'])
	for tweet in t_data:
		try:
			yield parse_tweet(tweet, t_includes, t_errors)
		except Exception as e:
			pprint(e)
			pprint(tweet)


def parse_tweet_file(file_path):
	tweets = read_file(file_path)
	parsed_tweets = []
	for tweet in parse_tweets(tweets):
		json_data = json.dumps(tweet)
		parsed_tweets.append(json_data)
	return parsed_tweets


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-ps', '--processes', default=8)
	args = parser.parse_args()

	files = [os.path.join(args.input_path, x) for x in os.listdir(args.input_path) if x.endswith('.json')]
	with open(args.output_path, 'w') as f:
		with Pool(processes=args.processes) as p:
			for tweets in tqdm(p.imap(parse_tweet_file, files), total=len(files)):
				for tweet in tweets:
					f.write(tweet + '\n')


if __name__ == '__main__':
	main()
