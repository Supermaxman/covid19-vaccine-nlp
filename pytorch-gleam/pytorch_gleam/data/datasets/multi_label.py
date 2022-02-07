
import json
from typing import List, Dict, Any, Union

import torch
from torch.utils.data import Dataset
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import MultiLabelBatchCollator


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


class MultiLabelDataset(Dataset):
	examples: List[Dict[Any, Union[Any, Dict]]]

	def __init__(self, data_path: Union[str, List[str]], label_name: str, tokenizer, label_map: Dict[str, int]):
		super().__init__()
		self.tokenizer = tokenizer
		self.label_name = label_name
		self.label_map = label_map

		self.examples = []
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
			ex_labels = []
			if self.label_name in ex:
				ex_labels = [self.label_map[label] for label in ex[self.label_name]]
			token_data = self.tokenizer(
				ex_text
			)
			example = {
				'ids': ex_id,
				'text': ex_text,
				'labels': ex_labels,
				'stage': stage,
				'input_ids': token_data['input_ids'],
				'attention_mask': token_data['attention_mask'],
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


class MultiLabelDataModule(BaseDataModule):
	def __init__(
			self,
			label_name: str,
			label_map: Dict[str, int],
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
		self.num_labels = len(label_map)
		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path
		self.predict_path = predict_path

		if self.train_path is not None:
			self.train_dataset = MultiLabelDataset(
				tokenizer=self.tokenizer,
				data_path=self.train_path,
				label_name=self.label_name,
				label_map=self.label_map
			)
		if self.val_path is not None:
			self.val_dataset = MultiLabelDataset(
				tokenizer=self.tokenizer,
				data_path=self.val_path,
				label_name=self.label_name,
				label_map=self.label_map
			)
		if self.test_path is not None:
			self.test_dataset = MultiLabelDataset(
				tokenizer=self.tokenizer,
				data_path=self.test_path,
				label_name=self.label_name,
				label_map=self.label_map
			)
		if self.predict_path is not None:
			self.predict_dataset = MultiLabelDataset(
				tokenizer=self.tokenizer,
				data_path=self.predict_path,
				label_name=self.label_name,
				label_map=self.label_map
			)

	def create_collator(self):
		return MultiLabelBatchCollator(
			max_seq_len=self.max_seq_len,
			use_tpus=self.use_tpus,
			num_labels=self.num_labels,
		)


