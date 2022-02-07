#!/usr/bin/env python

import argparse
import json
import pandas as pd


def parse_frames(frame_path):

	df = pd.read_excel(
		frame_path
	)
	frames = {}
	for idx, row in df.iterrows():
		f_id = row['f_id']
		frame = {
			'text': row['text'],
			'short_text': row['short_text'],
			'q_id': row['q_id'],
			'moralities': {
				key.strip().lower(): True for key in row['moralities'].split()
			}
		}
		frames[f_id] = frame
	return frames


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	frames = parse_frames(args.input_path)
	with open(args.output_path, 'w') as f:
		json.dump(frames, f, indent=2)


if __name__ == '__main__':
	main()


