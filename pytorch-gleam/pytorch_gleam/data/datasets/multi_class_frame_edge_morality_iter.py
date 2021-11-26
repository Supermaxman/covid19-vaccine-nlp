
import json
from typing import List, Dict, Any, Union
from itertools import islice, chain, zip_longest

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import MultiClassFrameEdgeMoralityBatchCollator
from tqdm import tqdm


def batch(iterable, n):
	args = [iter(iterable)] * n
	return (list(filter(lambda x: x is not None, x)) for x in zip_longest(fillvalue=None, *args))


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


class MultiClassFrameEdgeMoralityIterableDataset(IterableDataset):

	def __init__(
			self,
			batch_size: int,
			data_path: str,
			frame_path: str,
			label_name: str,
			size_estimate: int,
			morality_map: Dict[str, int],
	):
		super().__init__()
		self.morality_map = morality_map
		self.batch_size = batch_size
		self.frame_path = frame_path
		self.label_name = label_name
		self.size_estimate = size_estimate
		self.data_path = data_path

		with open(self.frame_path) as f:
			self.frames = json.load(f)

	def parse_example(self, ex):
		ex_id = ex['id']
		for f_id, f_label in ex[self.label_name].items():
			f_ex = ex['f_examples'][f_id]
			frame = self.frames[f_id]
			ex_label = 0
			ex_edges = {e_key: np.array(e_value) for e_key, e_value in f_ex['edges'].items()}
			ex_morality = []
			if 'morality_preds' in ex:
				ex_morality = [self.morality_map[m_name] for m_name in ex['morality_preds']]
			f_morality = []
			for m_name, m_val in frame['moralities'].items():
				if m_val:
					f_morality.append(self.morality_map[m_name])
			example = {
				'ids': f'{ex_id}|{f_id}',
				'label': ex_label,
				'input_ids': f_ex['input_ids'],
				'attention_mask': f_ex['attention_mask'],
				'edges': ex_edges,
				'ex_morality': ex_morality,
				'f_morality': f_morality,
			}
			if 'token_type_ids' in f_ex:
				example['token_type_ids'] = f_ex['token_type_ids']
			yield example

	def __len__(self):
		return self.size_estimate // self.batch_size

	def __iter__(self):
		iterator = self.ex_iter()
		for ex_batch in batch(iterator, self.batch_size):
			yield ex_batch

	def ex_iter(self):
		for ex in read_jsonl(self.data_path):
			for ex_example in self.parse_example(ex):
				yield ex_example

	def worker_init_fn(self, _):
		pass


class MultiClassFrameEdgeMoralityIterableDataModule(BaseDataModule):
	def __init__(
			self,
			label_name: str,
			label_map: Dict[str, int],
			morality_map: Dict[str, int],
			frame_path: str,
			predict_path: str,
			size_estimate: int,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.label_map = label_map
		self.morality_map = morality_map

		self.label_name = label_name
		self.predict_path = predict_path
		self.frame_path = frame_path
		self.size_estimate = size_estimate

		self.predict_dataset = MultiClassFrameEdgeMoralityIterableDataset(
			batch_size=self.batch_size,
			data_path=self.predict_path,
			frame_path=self.frame_path,
			label_name=self.label_name,
			morality_map=self.morality_map,
			size_estimate=self.size_estimate
		)

	def create_collator(self):
		return MultiClassFrameEdgeMoralityBatchCollator(
			max_seq_len=self.max_seq_len,
			use_tpus=self.use_tpus,
			num_moralities=len(self.morality_map)
		)


