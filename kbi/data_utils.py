
import json
from typing import Type
import random

from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch

from dm_utils import BaseDataModule


def get_data_module(data_type: str) -> Type[pl.LightningDataModule]:
	dm_map = {
		'mc_misinfo': MultiClassMisinfoDataModule
	}
	data_type = data_type.lower()
	if data_type not in dm_map:
		raise ValueError(f'Unknown data_type: {data_type}')
	return dm_map[data_type]


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


class MultiClassMisinfoDataset(Dataset):
	def __init__(self, tokenizer, data_path, misinfo_path):
		super().__init__()
		self.label_map = {
			'No Stance': 0,
			'no_stance': 0,
			'Accept': 1,
			'agree': 1,
			'Reject': 2,
			'disagree': 2
		}
		with open(misinfo_path) as f:
			self.misinfo = json.load(f)

		self.examples = []
		for ex in read_jsonl(data_path):
			if 'labels' in ex:
				ex_labels = ex['labels']
			elif 'misinfo' in ex:
				ex_labels = ex['misinfo']
			else:
				raise ValueError()
			for m_id, m_label in ex_labels.items():
				if m_id not in self.misinfo:
					continue
				if m_label not in self.label_map:
					continue
				m_label_idx = self.label_map[m_label]
				p_ex = self._create_example(tokenizer, ex, m_id, m_label_idx)
				self.examples.append(p_ex)
		random.shuffle(self.examples)

	def _create_example(self, tokenizer, ex, m_id, m_label):
		ex_id = ex['id']
		ex_text = ex['full_text'] if 'full_text' in ex else ex['text']
		ex_text = ex_text.strip().replace('\r', ' ').replace('\n', ' ')
		m = self.misinfo[m_id]
		m_text = m['text']
		token_data = tokenizer(
			m_text,
			ex_text
		)

		p_ex = {
			'id': f'{ex_id}|{m_id}',
			'labels': m_label,
			'input_ids': token_data['input_ids'],
			'token_type_ids': token_data['token_type_ids'],
			'attention_mask': token_data['attention_mask'],
		}
		return p_ex

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class MultiClassMisinfoDataModule(BaseDataModule):
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

		if self.train_path is not None and train_misinfo_path is not None:
			self.train_dataset = MultiClassMisinfoDataset(
				tokenizer=self.tokenizer,
				data_path=self.train_path,
				misinfo_path=train_misinfo_path
			)
		if self.val_path is not None and val_misinfo_path is not None:
			self.val_dataset = MultiClassMisinfoDataset(
				tokenizer=self.tokenizer,
				data_path=self.val_path,
				misinfo_path=val_misinfo_path
			)
		if self.test_path is not None and test_misinfo_path is not None:
			self.test_dataset = MultiClassMisinfoDataset(
				tokenizer=self.tokenizer,
				data_path=self.test_path,
				misinfo_path=test_misinfo_path
			)
		if self.predict_path is not None and predict_misinfo_path is not None:
			self.predict_dataset = MultiClassMisinfoDataset(
				tokenizer=self.tokenizer,
				data_path=self.predict_path,
				misinfo_path=predict_misinfo_path
			)
