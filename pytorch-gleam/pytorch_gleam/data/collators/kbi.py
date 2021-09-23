
import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class KbiBatchCollator(BatchCollator):
	def __init__(
			self,
			num_relations: int = 2,
			*args,
			**kwargs
	):
		self.num_relations = num_relations
		super().__init__(*args, **kwargs)

	def _calculate_multi_seq_padding(self, examples):
		if self.use_tpus:
			pad_seq_len = self.max_seq_len
		else:
			pad_seq_len = 0
			for ex in examples:
				ex_seqs = [ex['t_ex'], ex['m_ex']] + ex['p_samples'] + ex['n_samples']
				for ex_seq in ex_seqs:
					pad_seq_len = max(pad_seq_len, min(len(ex_seq['input_ids']), self.max_seq_len))
		return pad_seq_len

	def __call__(self, examples):
		pad_seq_len = self._calculate_multi_seq_padding(examples)
		num_examples = len(examples)
		pos_samples = len(examples[0]['p_samples'])
		neg_samples = len(examples[0]['n_samples'])
		num_samples = pos_samples + neg_samples
		num_sequences_per_example = 2 + pos_samples + neg_samples
		# ex + m + pos_samples + neg_samples
		num_sequences = num_examples * num_sequences_per_example

		input_ids = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
		direction_mask = torch.zeros([num_examples, 2], dtype=torch.float)
		labels = torch.zeros([num_examples, num_sequences_per_example - 1], dtype=torch.long)
		stages = torch.zeros([num_examples, num_sequences_per_example - 1], dtype=torch.long)
		relation_mask = torch.zeros([num_examples, num_samples, self.num_relations], dtype=torch.float)
		ids = []
		m_ids = []
		p_ids = []
		n_ids = []
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['t_ex']['t_id'])
			m_ids.append(ex['m_ex']['m_id'])
			for l_idx, r_label in enumerate(ex['relations']):
				relation_mask[ex_idx, l_idx, r_label] = 1.0
			for l_idx, label in enumerate(ex['labels']):
				labels[ex_idx, l_idx] = label
			for l_idx, stage in enumerate(ex['stages']):
				stages[ex_idx, l_idx] = stage
			ex_seqs = [ex['m_ex'], ex['t_ex']] + ex['p_samples'] + ex['n_samples']
			direction_mask[ex_idx, ex['direction']] = 1.0
			for seq_idx, seq in enumerate(ex_seqs):
				self.pad_and_apply_seq(seq['input_ids'], input_ids, ex_idx, seq_idx)
				self.pad_and_apply_seq(seq['attention_mask'], attention_mask, ex_idx, seq_idx)
				self.pad_and_apply_seq(seq['token_type_ids'], token_type_ids, ex_idx, seq_idx)
			for p_ex in ex['p_samples']:
				p_ids.append(p_ex['t_id'])
			for n_ex in ex['n_samples']:
				n_ids.append(n_ex['t_id'])
		batch = {
			'ids': ids,
			'm_ids': m_ids,
			'p_ids': p_ids,
			'num_examples': num_examples,
			'pos_samples': pos_samples,
			'neg_samples': neg_samples,
			'pad_seq_len': pad_seq_len,
			'num_sequences_per_example': num_sequences_per_example,
			'num_sequences': num_sequences,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'direction_mask': direction_mask,
			'labels': labels,
			'stages': stages,
			'relation_mask': relation_mask
		}

		return batch

	def pad_and_apply_seq(self, id_list, id_tensor, ex_idx, seq_idx):
		ex_ids = id_list[:self.max_seq_len]
		id_tensor[ex_idx, seq_idx, :len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)
