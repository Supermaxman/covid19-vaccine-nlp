
import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class NliTextBatchCollator(BatchCollator):
	def __init__(
			self,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)

	def __call__(self, examples):
		pad_seq_len = self._calculate_seq_padding(examples)
		num_examples = len(examples)

		input_ids = torch.zeros([num_examples, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([num_examples, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([num_examples, pad_seq_len], dtype=torch.long)
		labels = torch.zeros([num_examples, 2], dtype=torch.long)
		stages = torch.zeros([num_examples, 2], dtype=torch.long)
		relations = torch.zeros([num_examples], dtype=torch.long)
		ids = []
		m_ids = []
		p_ids = []
		has_token_type_ids = True
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['t_ex']['t_id'])
			m_ids.append(ex['m_ex']['m_id'])
			p_ids.append(ex['p_ex']['t_id'])
			relations[ex_idx] = ex['relations']
			for l_idx, label in enumerate(ex['labels']):
				labels[ex_idx, l_idx] = label
			for l_idx, stage in enumerate(ex['stages']):
				stages[ex_idx, l_idx] = stage
			self.pad_and_apply(ex['input_ids'], input_ids, ex_idx)
			self.pad_and_apply(ex['attention_mask'], attention_mask, ex_idx)
			if 'token_type_ids' in ex:
				self.pad_and_apply(ex['token_type_ids'], token_type_ids, ex_idx)
			else:
				has_token_type_ids = False

		batch = {
			'ids': ids,
			'm_ids': m_ids,
			'p_ids': p_ids,
			'num_examples': num_examples,
			'pad_seq_len': pad_seq_len,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'labels': labels,
			'stages': stages,
			'relations': relations
		}
		if has_token_type_ids:
			batch['token_type_ids'] = token_type_ids
		return batch
