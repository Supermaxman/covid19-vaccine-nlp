
import torch
from abc import ABC, abstractmethod


class BatchCollator(ABC):
	def __init__(
			self,
			max_seq_len: int = 512,
			use_tpus=False
	):
		self.max_seq_len = max_seq_len
		self.use_tpus = use_tpus

	def _calculate_seq_padding(self, examples):
		if self.use_tpus:
			pad_seq_len = self.max_seq_len
		else:
			pad_seq_len = 0
			for ex in examples:
				pad_seq_len = max(pad_seq_len, min(len(ex['input_ids']), self.max_seq_len))

		return pad_seq_len

	def pad_and_apply(self, id_list, id_tensor, ex_idx):
		ex_ids = id_list[:self.max_seq_len]
		id_tensor[ex_idx, :len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)

	@abstractmethod
	def __call__(self, examples: list) -> dict:
		pass

