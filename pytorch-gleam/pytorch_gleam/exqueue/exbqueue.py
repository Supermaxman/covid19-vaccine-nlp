#!/usr/bin/env python

import os
import argparse
from pytorch_gleam.exqueue.exqueue import ex_queue


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--batch_path', required=True)
	parser.add_argument('-s', '--script_path', default='ex/tt.sh')
	parser.add_argument('-qp', '--queue_path', default='~/.default_queue')
	args = parser.parse_args()

	batch_path = args.batch_path
	script = args.script_path
	queue_path = args.queue_path

	for ex_file in os.listdir(batch_path):
		ex_path = os.path.join(batch_path, ex_file)
		experiment = f'{script} {ex_path}'
		ex_queue(experiment, queue_path)


if __name__ == '__main__':
	main()
