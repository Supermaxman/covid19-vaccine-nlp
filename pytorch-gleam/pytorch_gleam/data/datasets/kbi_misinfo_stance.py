import torch
from collections import defaultdict

from pytorch_gleam.data.base_datasets import BaseDataModule
from pytorch_gleam.data.datasets.misinfo_stance import MisinfoStanceDataset
from pytorch_gleam.data.collators import MultiSequenceBatchCollator


class KbiMisinfoStanceDataset(MisinfoStanceDataset):
	def __init__(self, tokenizer, pos_samples: int = 1, neg_samples: int = 1, shuffle: bool = True, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.pos_samples = pos_samples
		self.neg_samples = neg_samples
		self.shuffle = shuffle
		for m_id, m in self.misinfo.items():
			m['token_data'] = tokenizer(
				m['text']
			)
		self.base_examples = self.examples
		self.examples = []
		self.pos_examples = defaultdict(lambda: defaultdict(list))
		self.neg_examples = defaultdict(lambda: defaultdict(list))
		ex_token_data = {}
		for pair_ex in self.base_examples:
			ex_id = pair_ex['ex_id']
			if ex_id in ex_token_data:
				token_data = ex_token_data[ex_id]
			else:
				token_data = tokenizer(
					pair_ex['ex_text']
				)
				ex_token_data[ex_id] = token_data

			m_id = pair_ex['m_id']
			m = self.misinfo[m_id]
			m_label = pair_ex['m_label']
			m_ex = {
				'm_id': m_id,
				'm_text': pair_ex['m_text'],
				'input_ids': m['token_data']['input_ids'],
				'token_type_ids': m['token_data']['token_type_ids'],
				'attention_mask': m['token_data']['attention_mask'],
			}
			t_ex = {
				't_id': ex_id,
				't_text': pair_ex['ex_text'],
				'input_ids': token_data['input_ids'],
				'token_type_ids': token_data['token_type_ids'],
				'attention_mask': token_data['attention_mask'],
			}
			self.examples.append((t_ex, m_ex, m_label))

	def __getitem__(self, idx):
		# TODO
		if torch.is_tensor(idx):
			idx = idx.tolist()

		t_ex, m_ex, m_label = self.examples[idx]
		m_id = m_ex['m_id']
		t_id = t_ex['t_id']

		pos_samples = self._sample(
			self.pos_examples[m_id],
			self.pos_samples
		)

		neg_samples = self._sample(
			self.neg_examples[m_id],
			self.neg_samples
		)

		subj_obj_sample = self._sample_subj_obj()

		ex = {
			't_ex': t_ex,
			'm_ex': m_ex,
			'label': m_label,
			'p_samples': pos_samples,
			'n_samples': neg_samples,
			'subj_obj_sample': subj_obj_sample
		}

		return ex

	def _sample(self, m_examples, m_count):
		samples = []
		if m_count <= 0:
			return samples
		m_s_indices = torch.randperm(
			n=len(m_examples),
			generator=self.generator
		).tolist()[:m_count]
		for s_idx in m_s_indices:
			samples.append(m_examples[s_idx])
		return samples

	def _sample_subj_obj(self):
		r = torch.rand(
			size=(1,),
			generator=self.generator
		).tolist()[0]
		if r < 0.5:
			return 0
		else:
			return 1

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
