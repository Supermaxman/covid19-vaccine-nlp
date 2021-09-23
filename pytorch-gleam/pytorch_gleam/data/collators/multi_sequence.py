
import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class MultiSequenceBatchCollator(BatchCollator):
	def __init__(
			self,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)

	def __call__(self, examples: list) -> dict:
		pad_seq_len = self._calculate_seq_padding(examples)

		batch_size = len(examples)
		input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)

		# [ex_count]
		labels = torch.zeros([len(examples)], dtype=torch.long)
		for ex_idx, ex in enumerate(examples):
			if 'labels' not in ex:
				break
			labels[ex_idx] = ex['labels']

		ids = []
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['ids'])
			self.pad_and_apply(ex['input_ids'], input_ids, ex_idx)
			self.pad_and_apply(ex['attention_mask'], attention_mask, ex_idx)
			self.pad_and_apply(ex['token_type_ids'], token_type_ids, ex_idx)

		batch = {
			'ids': ids,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels,
		}

		return batch

