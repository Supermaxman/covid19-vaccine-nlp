
import json
from collections import defaultdict
from typing import List, Dict, Any, Union

import numpy as np

from py_lex import EmoLex
import torch
from torch.utils.data import Dataset

from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import MultiClassFrameEdgeBatchCollator
import pytorch_gleam.data.datasets.senticnet5 as senticnet5


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


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
		num_semantic_hops, num_emotion_hops, num_lexical_hops,
		emotion_type, emolex, lex_edge_expanded
):
	seq_len = len(wpt_tokens['input_ids'])
	align_map, a_tokens = align_token_sequences(m_tokens, t_tokens, wpt_tokens)

	semantic_edges = defaultdict(set)
	emotion_edges = defaultdict(set)
	reverse_emotion_edges = defaultdict(set)
	lexical_edges = defaultdict(set)
	reverse_lexical_dep_edges = defaultdict(set)
	reverse_lexical_pos_edges = defaultdict(set)
	lexical_dep_edges = defaultdict(set)
	lexical_pos_edges = defaultdict(set)
	root_text = None
	r_map = defaultdict(set)
	t_map = {}
	for token in a_tokens:
		text = token['text'].lower()
		head = token['head'].lower()
		for wpt_idx in token['wpt_idxs']:
			t_map[wpt_idx] = text
			r_map[text].add(wpt_idx)
		pos = token['pos']
		dep = token['dep']
		reverse_lexical_dep_edges[dep].add(text)
		reverse_lexical_pos_edges[pos].add(text)
		lexical_dep_edges[text].add(dep)
		lexical_pos_edges[text].add(pos)
		# will be two roots with two sequences
		if dep == 'ROOT':
			root_text = text
		sentic = token['sentic']
		if sentic is not None:
			for sem in sentic['semantics']:
				semantic_edges[text].add(sem)
			for i in range(num_semantic_hops-1):
				semantic_edges[text] = sentic_expand(semantic_edges[text], [8, 9, 10, 11, 12])
			if emotion_type == 'senticnet':
				emotion_edges[text].add(sentic['primary_mood'])
				emotion_edges[text].add(sentic['secondary_mood'])
				reverse_emotion_edges[sentic['primary_mood']].add(text)
				reverse_emotion_edges[sentic['secondary_mood']].add(text)
			elif emotion_type == 'emolex':
				for emotion in emolex.categorize_token(text):
					emotion_edges[text].add(emotion)
					reverse_emotion_edges[emotion].add(text)
			else:
				raise ValueError(f'Invalid emotion type: {emotion_type}')
		# for emotion in [sentic['primary_mood'], sentic['secondary_mood']]:
		# 	emotion_edges[text] = emotion_edges[text].union(emotion_nodes[emotion])

		# for i in range(num_emotion_hops - 1):
		# 	new_emotions = sentic_expand(emotion_edges[text], [4, 5])
		# 	for emotion in new_emotions:
		# 		emotion_edges[text] = emotion_edges[text].union(emotion_nodes[emotion])

		lexical_edges[text].add(head)

	lexical_edges['[CLS]'].add(root_text)
	lexical_edges['[SEP]'].add(root_text)

	# text -> emotion node -> other text in sentence with same emotions
	for text in emotion_edges.keys():
		emotions = emotion_edges[text]
		emotion_edges[text] = emotion_edges[text].union(
			set(flatten(reverse_emotion_edges[emotion] for emotion in emotions))
		)
	if 'dep' in lex_edge_expanded:
		for text in lexical_edges.keys():
			# expand lexical edges to same dependency roles
			text_deps = lexical_dep_edges[text]
			lexical_edges[text] = lexical_edges[text].union(
				set(flatten(reverse_lexical_dep_edges[dep] for dep in text_deps))
			)

	if 'pos' in lex_edge_expanded:
		for text in lexical_edges.keys():
			# expand lexical edges to same pos tags
			text_pos = lexical_pos_edges[text]
			lexical_edges[text] = lexical_edges[text].union(
				set(flatten(reverse_lexical_pos_edges[pos] for pos in text_pos))
			)

	semantic_adj = create_adjacency_matrix(
		edges=semantic_edges,
		size=seq_len,
		t_map=t_map,
		r_map=r_map
	)
	emotion_adj = create_adjacency_matrix(
		edges=emotion_edges,
		size=seq_len,
		t_map=t_map,
		r_map=r_map
	)
	lexical_adj = create_adjacency_matrix(
		edges=lexical_edges,
		size=seq_len,
		t_map=t_map,
		r_map=r_map
	)

	edges = {
		'semantic': semantic_adj,
		'emotion': emotion_adj,
		'lexical': lexical_adj,
	}
	return edges


class MultiClassFrameEdgeDataset(Dataset):
	examples: List[Dict[Any, Union[Any, Dict]]]

	def __init__(
			self, data_path: Union[str, List[str]], frame_path: str,
			label_name: str, tokenizer, label_map: Dict[str, int],
			emo_path: str,
			num_semantic_hops: int = 3,
			num_emotion_hops: int = 1,
			num_lexical_hops: int = 1,
			emotion_type: str = 'senticnet',
			lex_edge_expanded: str = 'none',
	):
		super().__init__()
		self.frame_path = frame_path
		self.tokenizer = tokenizer
		self.label_name = label_name
		self.label_map = label_map

		self.num_semantic_hops = num_semantic_hops
		self.num_emotion_hops = num_emotion_hops
		self.num_lexical_hops = num_lexical_hops
		self.emotion_type = emotion_type
		self.lex_edge_expanded = lex_edge_expanded
		self.emolex = EmoLex(emo_path)

		self.examples = []
		with open(self.frame_path) as f:
			self.frames = json.load(f)
		if isinstance(data_path, str):
			self.read_path(data_path)
		else:
			for stage, stage_path in enumerate(data_path):
				self.read_path(stage_path, stage)

	def read_path(self, data_path, stage=0):
		for ex in read_jsonl(data_path):
			ex_id = ex['id']
			ex_text = ex['full_text'] if 'full_text' in ex else ex['text']
			ex_text = ex_text.strip().replace('\r', ' ').replace('\n', ' ')
			for f_id, f_label in ex[self.label_name].items():
				frame = self.frames[f_id]
				frame_text = frame['text']
				ex_label = 0
				if f_label in self.label_map:
					ex_label = self.label_map[f_label]
				token_data = self.tokenizer(
					frame_text,
					ex_text
				)

				tweet_parse = [add_sentic_token_features(x) for x in ex['parse']]
				f_parse = [add_sentic_token_features(x) for x in frame['parse']]

				ex_edges = create_edges(
					f_parse,
					tweet_parse,
					token_data,
					self.num_semantic_hops,
					self.num_emotion_hops,
					self.num_lexical_hops,
					self.emotion_type,
					self.emolex,
					self.lex_edge_expanded,
				)

				example = {
					'ids': f'{ex_id}|{f_id}',
					'label': ex_label,
					'input_ids': token_data['input_ids'],
					'attention_mask': token_data['attention_mask'],
					'edges': ex_edges
				}
				if 'token_type_ids' in token_data:
					example['token_type_ids'] = token_data['token_type_ids']

				self.examples.append(example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example

	def worker_init_fn(self, _):
		pass


class MultiClassFrameEdgeDataModule(BaseDataModule):
	def __init__(
			self,
			label_name: str,
			label_map: Dict[str, int],
			frame_path: str,
			emo_path: str,
			num_semantic_hops: int = 3,
			num_emotion_hops: int = 1,
			num_lexical_hops: int = 1,
			emotion_type: str = 'senticnet',
			lex_edge_expanded: str = 'none',
			train_path: str = None,
			val_path: str = None,
			test_path: str = None,
			predict_path: str = None,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.label_map = label_map

		self.label_name = label_name
		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path
		self.predict_path = predict_path
		self.frame_path = frame_path

		if self.train_path is not None:
			self.train_dataset = MultiClassFrameEdgeDataset(
				tokenizer=self.tokenizer,
				data_path=self.train_path,
				frame_path=self.frame_path,
				label_name=self.label_name,
				label_map=self.label_map,
				emo_path=emo_path,
				num_semantic_hops=num_semantic_hops,
				num_emotion_hops=num_emotion_hops,
				num_lexical_hops=num_lexical_hops,
				emotion_type=emotion_type,
				lex_edge_expanded=lex_edge_expanded,
			)
		if self.val_path is not None:
			self.val_dataset = MultiClassFrameEdgeDataset(
				tokenizer=self.tokenizer,
				data_path=self.val_path,
				frame_path=self.frame_path,
				label_name=self.label_name,
				label_map=self.label_map,
				emo_path=emo_path,
				num_semantic_hops=num_semantic_hops,
				num_emotion_hops=num_emotion_hops,
				num_lexical_hops=num_lexical_hops,
				emotion_type=emotion_type,
				lex_edge_expanded=lex_edge_expanded,
			)
		if self.test_path is not None:
			self.test_dataset = MultiClassFrameEdgeDataset(
				tokenizer=self.tokenizer,
				data_path=self.test_path,
				frame_path=self.frame_path,
				label_name=self.label_name,
				label_map=self.label_map,
				emo_path=emo_path,
				num_semantic_hops=num_semantic_hops,
				num_emotion_hops=num_emotion_hops,
				num_lexical_hops=num_lexical_hops,
				emotion_type=emotion_type,
				lex_edge_expanded=lex_edge_expanded,
			)
		if self.predict_path is not None:
			self.predict_dataset = MultiClassFrameEdgeDataset(
				tokenizer=self.tokenizer,
				data_path=self.predict_path,
				frame_path=self.frame_path,
				label_name=self.label_name,
				label_map=self.label_map,
				emo_path=emo_path,
				num_semantic_hops=num_semantic_hops,
				num_emotion_hops=num_emotion_hops,
				num_lexical_hops=num_lexical_hops,
				emotion_type=emotion_type,
				lex_edge_expanded=lex_edge_expanded,
			)

	def create_collator(self):
		return MultiClassFrameEdgeBatchCollator(
			max_seq_len=self.max_seq_len,
			use_tpus=self.use_tpus,
		)


