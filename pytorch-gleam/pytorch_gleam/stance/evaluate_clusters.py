
import argparse
import pickle
from collections import defaultdict
from textwrap import wrap
import random
import ujson as json
import math

import pandas as pd


def jaccard_coefficient(tp, tn, fp, fn):
	return tp / (tp + fn + fp)


def rand_index(tp, tn, fp, fn):
	return (tp + tn) / (tp + tn + fp + fn)


def fowlkes_mallow_index(tp, tn, fp, fn):
	return tp / math.sqrt((tp + fn) * (tp + fp))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-l', '--label_path', required=True)
	args = parser.parse_args()

	input_path = args.input_path
	label_path = args.label_path

	df = pd.read_excel(
		input_path,
		engine='openpyxl'
	)
	df = df.set_index('sample_id')

	with open(label_path, 'r') as f:
		labels = json.load(f)

	df['label'] = df.index.map(labels)

	df = df[df['Cluster'].notna()]
	df['judgement'] = df['Cluster'].apply(lambda x: 1 if x == 'Same' else 0)

	df['correct'] = df['judgement'] == df['label']
	df['incorrect'] = df['judgement'] != df['label']
	total_count = len(df)
	tp = (df['correct'] & (df['judgement'] == 1)).sum()
	tn = (df['correct'] & (df['judgement'] == 0)).sum()
	fp = (df['incorrect'] & (df['label'] == 1)).sum()
	fn = (df['incorrect'] & (df['label'] == 0)).sum()
	jc = jaccard_coefficient(tp, tn, fp, fn)
	ri = rand_index(tp, tn, fp, fn)
	fmi = fowlkes_mallow_index(tp, tn, fp, fn)

	print(f'jaccard_coefficient={jc:.3f}')
	print(f'rand_index={ri:.3f}')
	print(f'fowlkes_mallow_index={fmi:.3f}')


if __name__ == '__main__':
	main()
