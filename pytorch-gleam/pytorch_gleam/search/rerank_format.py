
import torch
import argparse
from collections import defaultdict
import os
import json


def load_predictions(input_path):
	pred_list = []
	for file_name in os.listdir(input_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(input_path, file_name))
			pred_list.extend(preds)

	question_scores = defaultdict(lambda: defaultdict(dict))
	p_count = 0
	for prediction in pred_list:
		doc_pass_id = prediction['id']
		q_p_id = prediction['question_id']
		# score = prediction['pos_score']
		score = prediction['pos_score'] - prediction['neg_score']
		question_scores[doc_pass_id][q_p_id] = score
		p_count += 1
	print(f'{p_count} total predictions')
	return question_scores


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	input_path = args.input_path
	output_path = args.output_path

	question_scores = load_predictions(input_path)
	with open(output_path, 'w') as f:
		json.dump(question_scores, f)


if __name__ == '__main__':
	main()
