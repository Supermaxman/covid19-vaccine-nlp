
import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class MultiClassFrameEdgeMoralityBatchCollator(BatchCollator):
	def __init__(
			self,
			num_moralities: int,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.num_moralities = num_moralities

	def __call__(self, examples: list) -> dict:
		print(f'b_examples={len(examples)}')
		if isinstance(examples[0], list) and len(examples) == 1:
			examples = examples[0]
		pad_seq_len = self._calculate_seq_padding(examples)

		batch_size = len(examples)
		input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)

		ex_morality = torch.zeros([batch_size, self.num_moralities], dtype=torch.long)
		f_morality = torch.zeros([batch_size, self.num_moralities], dtype=torch.long)

		edges = {}
		# [ex_count, num_classes]
		labels = torch.zeros([batch_size], dtype=torch.long)
		ids = []
		has_token_type_ids = True
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['ids'])
			self.pad_and_apply(ex['input_ids'], input_ids, ex_idx)
			self.pad_and_apply(ex['attention_mask'], attention_mask, ex_idx)
			if 'token_type_ids' in ex:
				self.pad_and_apply(ex['token_type_ids'], token_type_ids, ex_idx)
			else:
				has_token_type_ids = False
			if 'label' in ex:
				labels[ex_idx] = ex['label']
			for edge_name, edge_values in ex['edges'].items():
				# truncation to max_seq_len, still need to pad
				edge_values = edge_values[:pad_seq_len, :pad_seq_len]
				batch_edge_name = edge_name + '_edges'
				if batch_edge_name not in edges:
					edges[batch_edge_name] = torch.zeros([batch_size, pad_seq_len, pad_seq_len], dtype=torch.float)
				edges[batch_edge_name][ex_idx, :edge_values.shape[0], :edge_values.shape[1]] = torch.tensor(
					edge_values,
					dtype=torch.float
				)
			for m_idx in ex['ex_morality']:
				ex_morality[ex_idx, m_idx] = 1
			for m_idx in ex['f_morality']:
				f_morality[ex_idx, m_idx] = 1
		batch = {
			'ids': ids,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'ex_morality': ex_morality,
			'f_morality': f_morality,
			'labels': labels,
		}
		if has_token_type_ids:
			batch['token_type_ids'] = token_type_ids
			batch['f_seq_mask'] = (token_type_ids == 0).long()
			batch['ex_seq_mask'] = (token_type_ids == 1).long()
		else:
			batch['f_seq_mask'] = attention_mask
			batch['ex_seq_mask'] = attention_mask

		for edge_name, edge_value in edges.items():
			batch[edge_name] = edge_value
		return batch

