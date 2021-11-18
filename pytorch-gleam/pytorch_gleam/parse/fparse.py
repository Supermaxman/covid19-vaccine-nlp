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


def parse_frames(frame_path, nlp):
	with open(frame_path) as f:
		frames = json.load(f)
	for f_id, frame in tqdm(frames.items()):
		frame_text = frame['text']
		frame_parse = [get_token_features(x) for x in nlp(frame_text)]
		frame['parse'] = frame_parse
	return frames


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-m', '--model_name', default='en_core_web_sm')
	args = parser.parse_args()

	spacy_model = spacy.load(args.model_name)
	frames = parse_frames(args.input_path, spacy_model)
	with open(args.output_path, 'w') as f:
		json.dump(frames, f, indent=2)


if __name__ == '__main__':
	main()


