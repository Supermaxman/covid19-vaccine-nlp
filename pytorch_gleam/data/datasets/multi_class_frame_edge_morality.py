
import json
from typing import List, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import MultiClassFrameEdgeMoralityBatchCollator
from tqdm import tqdm


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


class MultiClassFrameEdgeMoralityDataset(Dataset):
	examples: List[Dict[Any, Union[Any, Dict]]]

	def __init__(
			self, data_path: Union[str, List[str]], frame_path: str,
			label_name: str, tokenizer, label_map: Dict[str, int],
			morality_map: Dict[str, int],
	):
		super().__init__()
		self.morality_map = morality_map
		self.frame_path = frame_path
		self.tokenizer = tokenizer
		self.label_name = label_name
		self.label_map = label_map

		self.examples = []
		with open(self.frame_path) as f:
			self.frames = json.load(f)
		if isinstance(data_path, str):
			self.read_path(data_path)
		else:
			for stage, stage_path in enumerate(data_path):
				self.read_path(stage_path, stage)

	def parse_example(self, ex):
		ex_id = ex['id']
		ex_examples = []
		for f_id, f_label in ex[self.label_name].items():
			f_ex = ex['f_examples'][f_id]
			frame = self.frames[f_id]
			ex_label = 0
			if f_label in self.label_map:
				ex_label = self.label_map[f_label]

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
			ex_examples.append(example)
		return ex_examples

	def read_path(self, data_path, stage=0):
		for ex_idx, ex in tqdm(enumerate(read_jsonl(data_path))):
			for ex_example in self.parse_example(ex):
				self.examples.append(ex_example)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example

	def worker_init_fn(self, _):
		pass


class MultiClassFrameEdgeMoralityDataModule(BaseDataModule):
	def __init__(
			self,
			label_name: str,
			label_map: Dict[str, int],
			morality_map: Dict[str, int],
			frame_path: str,
			train_path: str = None,
			val_path: str = None,
			test_path: str = None,
			predict_path: str = None,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.label_map = label_map
		self.morality_map = morality_map

		self.label_name = label_name
		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path
		self.predict_path = predict_path
		self.frame_path = frame_path

		if self.train_path is not None:
			self.train_dataset = MultiClassFrameEdgeMoralityDataset(
				tokenizer=self.tokenizer,
				data_path=self.train_path,
				frame_path=self.frame_path,
				label_name=self.label_name,
				label_map=self.label_map,
				morality_map=self.morality_map,
			)
		if self.val_path is not None:
			self.val_dataset = MultiClassFrameEdgeMoralityDataset(
				tokenizer=self.tokenizer,
				data_path=self.val_path,
				frame_path=self.frame_path,
				label_name=self.label_name,
				label_map=self.label_map,
				morality_map=self.morality_map,
			)
		if self.test_path is not None:
			self.test_dataset = MultiClassFrameEdgeMoralityDataset(
				tokenizer=self.tokenizer,
				data_path=self.test_path,
				frame_path=self.frame_path,
				label_name=self.label_name,
				label_map=self.label_map,
				morality_map=self.morality_map,
			)
		if self.predict_path is not None:
			self.predict_dataset = MultiClassFrameEdgeMoralityDataset(
				tokenizer=self.tokenizer,
				data_path=self.predict_path,
				frame_path=self.frame_path,
				label_name=self.label_name,
				label_map=self.label_map,
				morality_map=self.morality_map,
			)

	def create_collator(self):
		return MultiClassFrameEdgeMoralityBatchCollator(
			max_seq_len=self.max_seq_len,
			use_tpus=self.use_tpus,
			num_moralities=len(self.morality_map)
		)


