
import json

import torch

from pytorch_gleam.data.datasets.misinfo_stance import MisinfoStanceDataset
from pytorch_gleam.data.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import MultiSequenceBatchCollator


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


class MultiClassMisinfoStanceDataset(MisinfoStanceDataset):
	def __init__(self, tokenizer, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.base_examples = self.examples
		self.examples = []
		for ex in self.base_examples:
			ex_id = ex['ex_id']
			m_id = ex['m_id']
			token_data = tokenizer(
				ex['m_text'],
				ex['ex_text']
			)
			p_ex = {
				'ids': f'{ex_id}|{m_id}',
				'labels': ex['m_label'],
				'input_ids': token_data['input_ids'],
				'token_type_ids': token_data['token_type_ids'],
				'attention_mask': token_data['attention_mask'],
			}
			self.examples.append(p_ex)


class MultiClassMisinfoStanceDataModule(BaseDataModule):
	def __init__(
			self,
			train_misinfo_path: str = None,
			val_misinfo_path: str = None,
			test_misinfo_path: str = None,
			predict_misinfo_path: str = None,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)

		self.train_misinfo_path = train_misinfo_path
		self.val_misinfo_path = val_misinfo_path
		self.test_misinfo_path = test_misinfo_path
		self.predict_misinfo_path = predict_misinfo_path

		if self.train_path is not None and self.train_misinfo_path is not None:
			self.train_dataset = MultiClassMisinfoStanceDataset(
				tokenizer=self.tokenizer,
				data_path=self.train_path,
				misinfo_path=train_misinfo_path
			)
		if self.val_path is not None and self.val_misinfo_path is not None:
			self.val_dataset = MultiClassMisinfoStanceDataset(
				tokenizer=self.tokenizer,
				data_path=self.val_path,
				misinfo_path=val_misinfo_path
			)
		if self.test_path is not None and self.test_misinfo_path is not None:
			self.test_dataset = MultiClassMisinfoStanceDataset(
				tokenizer=self.tokenizer,
				data_path=self.test_path,
				misinfo_path=test_misinfo_path
			)
		if self.predict_path is not None and self.predict_misinfo_path is not None:
			self.predict_dataset = MultiClassMisinfoStanceDataset(
				tokenizer=self.tokenizer,
				data_path=self.predict_path,
				misinfo_path=predict_misinfo_path
			)

	def create_collator(self):
		return MultiSequenceBatchCollator(
			max_seq_len=self.max_seq_len,
			use_tpus=self.use_tpus,
		)
