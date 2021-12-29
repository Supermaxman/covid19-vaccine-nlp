#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import pytorch_gleam.data.datasets.senticnet5 as senticnet5
import spacy

tokenizer = None
nlp = None
num_semantic_hops: int = 3
label_name: str = 'labels'
frames: dict = {}


def get_sentic(word_text):
	word_text = word_text.lower()
	if word_text == 'coronavirus' or word_text == 'covid-19' or word_text == 'covid' or word_text == 'covid19':
		word_text = 'virus'
	if word_text not in senticnet5.senticnet:
		word_text = word_text[:-1]
		if word_text not in senticnet5.senticnet:
			word_text = word_text[:-1]
			if word_text not in senticnet5.senticnet:
				return None
	p_v, a_v, s_v, ap_v, p_m, s_m, po_l, po_v, s1, s2, s3, s4, s5 = senticnet5.senticnet[word_text]
	return {
		'pleasantness_value': float(p_v),
		'attention_value': float(a_v),
		'sensitivity_value': float(s_v),
		'aptitude_value': float(ap_v),
		'primary_mood': p_m,
		'secondary_mood': s_m,
		'polarity_label': po_l,
		'polarity_value': float(po_v),
		'semantics': [s1, s2, s3, s4, s5],
	}


def add_sentic_token_features(token_data):
	token_data = token_data.copy()
	sentic = get_sentic(token_data['text'])
	token_data['sentic'] = sentic
	return token_data


def align_tokens(tokens, wpt_tokens, seq_offset=0):
	align_map = {}
	for token in tokens:
		token['wpt_idxs'] = set()
		start = token['start']
		end = token['end']
		for char_idx in range(start, end):
			sub_token_idx = wpt_tokens.char_to_token(char_idx, sequence_index=seq_offset)
			# White spaces have no token and will return None
			if sub_token_idx is not None:
				align_map[sub_token_idx] = token
				token['wpt_idxs'].add(sub_token_idx)
	return align_map


def align_token_sequences(m_tokens, t_tokens, wpt_tokens):
	m_align_map = align_tokens(m_tokens, wpt_tokens)
	t_align_map = align_tokens(t_tokens, wpt_tokens, seq_offset=1)
	align_map = {**m_align_map, **t_align_map}
	aligned_tokens = []
	for sub_token_idx in range(len(wpt_tokens['input_ids'])):
		if sub_token_idx not in align_map:
			# CLS, SEP, or other special token
			aligned_token = {
				'pos': 'NONE',
				'dep': 'NONE',
				'head': 'NONE',
				'sentic': None,
				'text': '[CLS]' if sub_token_idx == 0 else '[SEP]',
				'wpt_idxs': {sub_token_idx}
			}
			align_map[sub_token_idx] = aligned_token
		aligned_token = align_map[sub_token_idx]
		aligned_tokens.append(aligned_token)

	return align_map, aligned_tokens


def flatten(multi_list):
	return [item for sub_list in multi_list for item in sub_list]


def create_adjacency_matrix(edges, size, t_map, r_map):
	adj = np.eye(size, dtype=np.float32)
	for input_idx in range(size):
		input_idx_text = t_map[input_idx]
		i_edges = set(flatten([r_map[e_txt] for e_txt in edges[input_idx_text]]))
		for edge_idx in i_edges:
			adj[input_idx, edge_idx] = 1.0
			adj[edge_idx, input_idx] = 1.0
	return adj


def sentic_expand(sentic_edges, expand_list):
	new_edges = set(sentic_edges)
	for edge in sentic_edges:
		edge_info = senticnet5.senticnet[edge]
		for i in expand_list:
			new_edges.add(edge_info[i])
	return new_edges


def create_edges(
		m_tokens, t_tokens, wpt_tokens,
		num_semantic_hops
):
	seq_len = len(wpt_tokens['input_ids'])
	align_map, a_tokens = align_token_sequences(m_tokens, t_tokens, wpt_tokens)

	semantic_edges = defaultdict(set)
	reverse_emotion_edges = defaultdict(set)
	lexical_edges = defaultdict(set)
	root_text = None
	r_map = defaultdict(set)
	t_map = {}

	for token in a_tokens:
		text = token['text'].lower()
		head = token['head'].lower()
		for wpt_idx in token['wpt_idxs']:
			t_map[wpt_idx] = text
			r_map[text].add(wpt_idx)
		# pos = token['pos']
		dep = token['dep']
		# will be two roots with two sequences
		if dep == 'ROOT':
			root_text = text
		sentic = token['sentic']
		if sentic is not None:
			for sem in sentic['semantics']:
				semantic_edges[text].add(sem)
			for i in range(num_semantic_hops-1):
				semantic_edges[text] = sentic_expand(semantic_edges[text], [8, 9, 10, 11, 12])
			reverse_emotion_edges[sentic['primary_mood']].add(text)
			reverse_emotion_edges[sentic['secondary_mood']].add(text)

		lexical_edges[text].add(head)
	lexical_edges['[CLS]'].add(root_text)
	lexical_edges['[SEP]'].add(root_text)

	# text -> emotion node -> other text in sentence with same emotions
	emotion_edges = defaultdict(set)
	for emotion, texts in reverse_emotion_edges.items():
		for text in texts:
			emotion_edges[text] = texts

	semantic_adj = np.eye(seq_len, dtype=np.float32)
	emotion_adj = np.eye(seq_len, dtype=np.float32)
	lexical_adj = np.eye(seq_len, dtype=np.float32)
	for input_idx_text, input_indices in r_map.items():
		input_indices = list(input_indices)
		for e_txt in semantic_edges[input_idx_text]:
			if e_txt in r_map:
				r_indices = list(r_map[e_txt])
				for idx in input_indices:
					for r_idx in r_indices:
						semantic_adj[idx, r_idx] = 1.0
						semantic_adj[r_idx, idx] = 1.0
		for e_txt in emotion_edges[input_idx_text]:
			if e_txt in r_map:
				r_indices = list(r_map[e_txt])
				for idx in input_indices:
					for r_idx in r_indices:
						emotion_adj[idx, r_idx] = 1.0
						emotion_adj[r_idx, idx] = 1.0
		for e_txt in lexical_edges[input_idx_text]:
			if e_txt in r_map:
				r_indices = list(r_map[e_txt])
				for idx in input_indices:
					for r_idx in r_indices:
						lexical_adj[idx, r_idx] = 1.0
						lexical_adj[r_idx, idx] = 1.0

	edges = {
		'semantic': semantic_adj.tolist(),
		'emotion': emotion_adj.tolist(),
		'lexical': lexical_adj.tolist(),
	}
	return edges


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


def parse_tweet(ex: dict):
	ex_text = ex['full_text'] if 'full_text' in ex else ex['text']
	ex_text = ex_text.strip().replace('\r', ' ').replace('\n', ' ')
	tweet_parse = [get_token_features(x) for x in nlp(ex_text)]
	ex['parse'] = tweet_parse
	ex_f_examples = {}
	for f_id, f_label in ex[label_name].items():
		frame = frames[f_id]
		frame_text = frame['text']
		token_data = tokenizer(
			frame_text,
			ex_text
		)

		tweet_parse = [add_sentic_token_features(x) for x in ex['parse']]
		f_parse = [add_sentic_token_features(x) for x in frame['parse']]

		ex_edges = create_edges(
			f_parse,
			tweet_parse,
			token_data,
			num_semantic_hops
		)
		f_example = {
			'input_ids': token_data['input_ids'],
			'attention_mask': token_data['attention_mask'],
			'edges': ex_edges,
		}
		if 'token_type_ids' in token_data:
			f_example['token_type_ids'] = token_data['token_type_ids']
		ex_f_examples[f_id] = f_example
	ex['f_examples'] = ex_f_examples
	return ex


def parse_tweets(tweet_path: str, num_processes: int):
	# for ex in tqdm(read_jsonl(tweet_path)):
	with Pool(processes=num_processes) as p:
		for ex in tqdm(
			p.imap_unordered(parse_tweet, read_jsonl(tweet_path)),
			miniters=10000,
			total=8161354
		):
			yield ex


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-f', '--frame_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-l', '--label_name', default='candidates')
	parser.add_argument('-t', '--tokenizer', default='digitalepidemiologylab/covid-twitter-bert-v2')
	parser.add_argument('-m', '--model_name', default='en_core_web_sm')
	parser.add_argument('-sh', '--num_semantic_hops', default=3)
	parser.add_argument('-p', '--num_processes', default=8)
	args = parser.parse_args()
	# for multiprocessing pool
	global tokenizer
	global nlp
	global frames
	global num_semantic_hops
	global label_name

	num_semantic_hops = args.num_semantic_hops
	label_name = args.label_name

	with open(args.frame_path) as f:
		frames = json.load(f)

	nlp = spacy.load(args.model_name)
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
	write_jsonl(
		parse_tweets(args.input_path, args.num_processes),
		args.output_path
	)


if __name__ == '__main__':
	main()


