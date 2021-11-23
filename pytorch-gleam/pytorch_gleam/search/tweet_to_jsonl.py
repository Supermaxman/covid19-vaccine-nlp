import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					ex = json.loads(line)
					yield ex
				except Exception as e:
					print(e)


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


def get_tweet_text(tweet):
	tweet_text = tweet['text']
	if tweet_text.startswith('RT '):
		ref_tweets = tweet['referenced_tweets']
		if len(ref_tweets) > 0:
			rt_data = ref_tweets[0]['data']
			if 'text' in rt_data:
				tweet_text = rt_data['text']
	if 'entities' in tweet:
		for e_type, e_list in tweet['entities'].items():
			if e_type == 'urls':
				for e_url in e_list:
					r_url = e_url['url']
					s_url = e_url['expanded_url']
					tweet_text = tweet_text.replace(r_url, s_url)
	return tweet_text


def invert_errors(errors):
	inv = defaultdict(dict)
	for error in errors:
		if 'resource_id' in error:
			r_id = error['resource_id']
		else:
			r_id = error['value']
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
	tweet = tweets['data']
	if 'includes' not in tweets:
		tweets['includes'] = {}
	if 'errors' not in tweets:
		tweets['errors'] = []
	t_includes = invert_includes(tweets['includes'])
	t_errors = invert_errors(tweets['errors'])
	return parse_tweet(tweet, t_includes, t_errors)


def create_jsonl_doc(tweet):
	if 'data' in tweet:
		tweet = parse_tweets(tweet)
	tweet_id = tweet['id']
	tweet_text = get_tweet_text(tweet)
	doc = {
		'id': tweet_id,
		'contents': tweet_text,
		'tweet': tweet
	}
	return doc


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	print('Loading tweets...')
	tweets = read_jsonl(args.input_path)

	print('Writing jsonl tweets...')
	write_jsonl(
		tqdm(
			(
				create_jsonl_doc(tweet) for tweet in tweets
			)
		),
		args.output_path
	)

	print('Done!')


if __name__ == '__main__':
	main()
