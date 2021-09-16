
import json

import torch
from torch.utils.data import Dataset


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				yield ex


class MisinfoStanceDataset(Dataset):
	def __init__(self, data_path, misinfo_path):
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
			ex_id = ex['id']
			ex_text = ex['full_text'] if 'full_text' in ex else ex['text']
			ex_text = ex_text.strip().replace('\r', ' ').replace('\n', ' ')
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
				m = self.misinfo[m_id]
				m_text = m['text']
				example = {
					'ex_id': ex_id,
					'm_id': m_id,
					'ex_text': ex_text,
					'm_text': m_text,
					'm_label': m_label_idx
				}
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


