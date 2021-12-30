
from typing import List, Dict, Any, Union
from itertools import islice, chain, zip_longest

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.distributed as dist

from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import MultiClassFrameEdgeMoralityBatchCollator
from tqdm import tqdm
import ujson as json


def batch(iterable, n):
	try:
		while True:
			batch_iter = list(islice(iterable, n))
			yield batch_iter
	except StopIteration:
		return


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


def worker_init_fn(_):
	process_id = dist.get_rank()
	num_processes = dist.get_world_size()

	worker_info = torch.utils.data.get_worker_info()
	worker_id = worker_info.id
	num_workers = worker_info.num_workers
	print(f'INFO: WORKER_INIT WORKER_INFO: {worker_id}/{num_workers}')
	print(f'INFO: WORKER_INIT: RANK_INFO: {process_id}/{num_processes}')
	dataset = worker_info.dataset
	dataset.frequency = (process_id * num_workers) + worker_id
	dataset.num_workers = num_processes * num_workers
	print(f'INFO: WORKER_INIT: F_INFO: {dataset.frequency}/{dataset.num_workers}')


class MultiClassFrameEdgeMoralityIterableDataset(IterableDataset):
	def __init__(
			self,
			tokenizer,
			data_path: str,
			frame_path: str,
			label_name: str,
			morality_map: Dict[str, int],
			worker_estimate: int,
			size_estimate: int,
	):
		super().__init__()
		self.tokenizer = tokenizer
		self.morality_map = morality_map
		self.frame_path = frame_path
		self.label_name = label_name
		self.data_path = data_path

		self.frequency = 0
		self.num_workers = 1
		self.worker_estimate = worker_estimate
		self.size_estimate = size_estimate

		with open(self.frame_path) as f:
			self.frames = json.load(f)

		self.num_examples = self.size_estimate
		# for ex in read_jsonl(self.data_path):
		# 	self.num_examples += len(ex[self.label_name])

		print(f'Num examples: {self.num_examples}')

	def __len__(self):
		return self.num_examples // self.worker_estimate

	def __iter__(self):
		ex_idx = 0
		for tweet in read_jsonl(self.data_path):
			ex_id = tweet['id']
			ex_text = tweet['full_text'] if 'full_text' in tweet else tweet['text']
			ex_text = ex_text.strip().replace('\r', ' ').replace('\n', ' ')

			for f_id, f_label in tweet[self.label_name].items():
				if ex_idx % self.num_workers == self.frequency:
					f_ex = tweet['f_examples'][f_id]
					frame = self.frames[f_id]
					frame_text = frame['text']
					token_data = self.tokenizer(
						frame_text,
						ex_text
					)
					seq_len = len(token_data['input_ids'])
					ex_label = 0
					# create np.eye of floats, then fill in with tuples
					# ex_edges = {e_key: np.array(e_value) for e_key, e_value in f_ex['edges'].items()}
					ex_edges = {}
					for edge_type, edge_list in f_ex['edges'].items():
						adj_list = np.eye(seq_len, dtype=np.float32)
						for i, j in edge_list:
							adj_list[i, j] = 1.0
							adj_list[j, i] = 1.0
						ex_edges[edge_type] = adj_list.tolist()
					ex_morality = []
					if 'morality_preds' in tweet:
						ex_morality = [self.morality_map[m_name] for m_name in tweet['morality_preds']]
					f_morality = []
					for m_name, m_val in frame['moralities'].items():
						if m_val:
							f_morality.append(self.morality_map[m_name])
					example = {
						'ids': f'{ex_id}|{f_id}',
						'label': ex_label,
						'input_ids': token_data['input_ids'],
						'attention_mask': token_data['attention_mask'],
						'edges': ex_edges,
						'ex_morality': ex_morality,
						'f_morality': f_morality,
					}
					if 'token_type_ids' in token_data:
						example['token_type_ids'] = token_data['token_type_ids']
					yield example
				ex_idx += 1

	def worker_init_fn(self, _):
		return worker_init_fn(_)


class MultiClassFrameEdgeMoralityIterableDataModule(BaseDataModule):
	def __init__(
			self,
			label_name: str,
			label_map: Dict[str, int],
			morality_map: Dict[str, int],
			frame_path: str,
			predict_path: str,
			worker_estimate: int,
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
		self.worker_estimate = worker_estimate
		self.size_estimate = size_estimate

		self.predict_dataset = MultiClassFrameEdgeMoralityIterableDataset(
			tokenizer=self.tokenizer,
			data_path=self.predict_path,
			frame_path=self.frame_path,
			label_name=self.label_name,
			morality_map=self.morality_map,
			worker_estimate=self.worker_estimate,
			size_estimate=self.size_estimate,
		)

	def create_collator(self):
		return MultiClassFrameEdgeMoralityBatchCollator(
			max_seq_len=self.max_seq_len,
			use_tpus=self.use_tpus,
			num_moralities=len(self.morality_map)
		)


