
import argparse

import ujson as json
from tqdm import tqdm


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


def write_jsonl(file_path, data):
	with open(file_path, 'w') as f:
		for row in data:
			f.write(json.dumps(row) + '\n')


def strip_annotations(tweet, label_name):
	ann = {
		'id': tweet['id'],
		label_name: tweet[label_name]
	}
	return ann


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-l', '--label_name', default='labels')
	args = parser.parse_args()

	input_path = args.input_path
	output_path = args.output_path
	label_name = args.label_name

	write_jsonl(
		output_path,
		tqdm(
			strip_annotations(tweet, label_name)
			for tweet in read_jsonl(input_path)
		)
	)


if __name__ == '__main__':
	main()
