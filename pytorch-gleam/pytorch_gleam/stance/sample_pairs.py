
import argparse
import pickle
from collections import defaultdict
from textwrap import wrap
import random
import ujson as json
import string

import pandas as pd


def create_user_text(u_tweets):
	u_info = u_tweets[0]['author']
	lines = [f'@{u_info["username"]}']
	for tweet in u_tweets:
		tweet_text = tweet['full_text'] if 'full_text' in tweet else tweet['text']
		for line in wrap(tweet_text, 50):
			lines.append(f'    {line}')
		lines.append(f'  ----')
	return '\n'.join(lines)


def sample_from_pop(pos_pop, sample_pop, seen_sample_ids, num_samples):
	samples = []
	while len(samples) < num_samples:
		pos_user = random.sample(pos_pop, 1)[0]
		pos_user_id = pos_user['user_id']
		pos_user_tweets = pos_user['tweets']
		neg_user = random.sample(sample_pop, 1)[0]
		neg_user_id = neg_user['user_id']
		neg_user_tweets = neg_user['tweets']
		sample_id = frozenset([pos_user_id, neg_user_id])
		if sample_id in seen_sample_ids or pos_user_id == neg_user_id:
			continue
		samples.append({
			'sample_id': f'{pos_user_id}|{neg_user_id}',
			'user_a_id': pos_user_id,
			'user_b_id': neg_user_id,
			'user_a': create_user_text(pos_user_tweets),
			'user_b': create_user_text(neg_user_tweets)
		})
	return samples


def create_excel(data, output_path, columns=None, labels=None):
	labels = labels.copy()
	if columns is None:
		columns = list(data[0].keys())
	column_types = {}
	for label_letter, label_name in zip(string.ascii_uppercase, labels):
		columns.append(label_name)
		labels[label_name]['label_letter'] = label_letter
	# TODO support more than A-Z columns
	for column_letter, column_name in zip(string.ascii_uppercase, columns):
		if column_name == 'id' or column_name.endswith('_id'):
			column_types[column_letter] = 'id'
		elif column_name in labels:
			labels[column_name]['data_letter'] = column_letter
			column_types[column_letter] = 'label'
		else:
			column_types[column_letter] = 'text'

	df = pd.DataFrame(
		data=data,
		columns=columns
	)
	df = df.fillna('')
	df = df.set_index(columns[0])
	data_size = len(df)
	l_df = pd.DataFrame({
		l_name: l['values'] for l_name, l in labels.items()
	})

	writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
	df.to_excel(writer, sheet_name='data')
	l_df.to_excel(writer, sheet_name='labels', index=False)
	workbook = writer.book
	worksheet = writer.sheets['data']
	for label_name, label in labels.items():
		data_letter = label['data_letter']
		label_letter = label['label_letter']
		label_values = label['values']
		label_message = label['message']
		worksheet.data_validation(
			f'{data_letter}2:{data_letter}{data_size+1}',
			{
				'validate': 'list',
				'source': f'=labels!${label_letter}$2:${label_letter}${len(label_values)+1}',
				'input_message': label_message
			}
		)
	cell_format = workbook.add_format()
	cell_format.set_text_wrap()
	for column_letter, column_type in column_types.items():
		column_range = f'{column_letter}:{column_letter}'
		# first_col, last_col, width, cell_format, options
		if column_type == 'id':
			worksheet.set_column(
				column_range,
				options={'hidden': True}
			)
		elif column_type == 'text':
			worksheet.set_column(
				column_range,
				cell_format=cell_format,
				width=55
			)
		elif column_type == 'label':
			worksheet.set_column(
				column_range,
				width=15
			)

	writer.save()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-s', '--samples_per_cluster', type=int, default=20)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-l', '--label_path', required=True)
	args = parser.parse_args()
	random.seed(0)
	input_path = args.input_path
	samples_per_cluster = args.samples_per_cluster
	output_path = args.output_path
	label_path = args.label_path

	with open(input_path, 'rb') as f:
		cluster_samples = pickle.load(f)
	cluster_ids = list(cluster_samples.keys())
	negative_samples = defaultdict(list)
	for cluster_a_id in cluster_ids:
		for cluster_b_id, cluster_s in cluster_samples.items():
			if cluster_a_id == cluster_b_id:
				continue
			negative_samples[cluster_a_id].extend(cluster_s)

	seen_sample_ids = set()
	labels = {}
	samples = []
	for cluster_a_id in cluster_ids:
		pos_pop = cluster_samples[cluster_a_id]
		neg_pop = negative_samples[cluster_a_id]
		pos_samples = sample_from_pop(pos_pop, pos_pop, seen_sample_ids, samples_per_cluster)
		for ps in pos_samples:
			labels[ps['sample_id']] = 1
		samples.extend(pos_samples)
		neg_samples = sample_from_pop(pos_pop, neg_pop, seen_sample_ids, samples_per_cluster)
		for ns in neg_samples:
			labels[ns['sample_id']] = 0
		samples.extend(neg_samples)

	random.shuffle(samples)
	create_excel(
		samples,
		output_path,
		labels={
			'Cluster': {
				'values': ['Same', 'Different'],
				'message': 'Identify if the two users are from the same or different cluster: '
			}
		}
	)


	with open(label_path, 'w') as f:
		json.dump(labels, f)



if __name__ == '__main__':
	main()
