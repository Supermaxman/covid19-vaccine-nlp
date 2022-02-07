#!/usr/bin/env python

import argparse
import json

import spacy
from tqdm import tqdm


def get_token_features(token):
	token_data = {
		'text': token.text,
		'pos': token.pos_,
		'dep': token.dep_,
		'head': token.head.text,
		'start': token.idx,
		'end': token.idx + len(token.text),
	}
	return token_data


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


def parse_tweets(tweet_path, nlp):
	for ex in tqdm(read_jsonl(tweet_path)):
		ex_text = ex['full_text'] if 'full_text' in ex else ex['text']
		ex_text = ex_text.strip().replace('\r', ' ').replace('\n', ' ')
		tweet_parse = [get_token_features(x) for x in nlp(ex_text)]
		ex['parse'] = tweet_parse
		yield ex


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-m', '--model_name', default='en_core_web_sm')
	args = parser.parse_args()

	spacy_model = spacy.load(args.model_name)
	write_jsonl(
		parse_tweets(args.input_path, spacy_model),
		args.output_path
	)


if __name__ == '__main__':
	main()


