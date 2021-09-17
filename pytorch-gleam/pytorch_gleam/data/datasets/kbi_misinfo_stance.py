import torch
from collections import defaultdict

from pytorch_gleam.data.base_datasets import BaseDataModule
from pytorch_gleam.data.datasets.misinfo_stance import MisinfoStanceDataset
from pytorch_gleam.data.collators import KbiBatchCollator


class KbiMisinfoStanceDataset(MisinfoStanceDataset):
	def __init__(self, tokenizer, pos_samples: int = 1, neg_samples: int = 1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.pos_samples = pos_samples
		self.neg_samples = neg_samples
		self.permutations = [
			self.flip_polarity,
			self.flip_rel,
			self.zero_polarity,
			# self.zero_all_polarity
		]
		for m_id, m in self.misinfo.items():
			m['token_data'] = tokenizer(
				m['text']
			)
		self.base_examples = self.examples
		self.examples = []
		self.label_examples = defaultdict(lambda: defaultdict(list))
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
			ex = {
				't_ex': t_ex,
				'm_ex': m_ex,
				'm_label': m_label
			}
			self.label_examples[m_id][m_label].append(ex)
			# no stance has no true pairs
			if m_label != 0:
				self.examples.append(ex)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		ex = self.examples[idx]
		t_ex = ex['t_ex']
		m_ex = ex['m_ex']
		# 1 is accept
		# 2 is reject
		# 0 is no stance, and is only in negative_examples
		tm_stance = ex['m_label']
		m_id = m_ex['m_id']
		t_id = t_ex['t_id']

		# TODO positive examples
		# 0 is entail
		# 1 is contradict
		# TODO could sample positive relations too
		tmp_relation = self._sample_relation()
		pm_stance = self.tmp_stance(tmp_relation, tm_stance)

		pos_examples = self.label_examples[m_id][pm_stance]
		# if pos_examples is empty then flip tmp_relation
		if len(pos_examples) == 0:
			tmp_relation = (tmp_relation + 1) % 2
			pm_stance = self.tmp_stance(tmp_relation, tm_stance)
			pos_examples = self.label_examples[m_id][pm_stance]

		# t - tmp_relation -> p
		pos_samples = self._sample(
			pos_examples,
			self.pos_samples,
			replacement=False
		)
		# negative sampling
		# Four ways to create negative samples
		# (flip polarity): a -> a to a -> r
		# (flip rel) a -> a to a \-> a
		# (zero polarity): a -> a to a -> ns
		# (zero all polarity) a -> a to ns -> ns
		# all four permutations could be useful for training
		# first three are necessary
		# fourth is only needed if ns will be used in seed kb for inference
		neg_relation_samples = self._negative_sample(
			m_id,
			tmp_relation,
			pm_stance,
			pos_samples,
			self.neg_samples
		)
		neg_samples = []
		neg_relations = []
		for neg_relation, neg_sample in neg_relation_samples:
			neg_samples.append(neg_sample)
			neg_relations.append(neg_relation)

		direction = self._sample_direction()
		# [pos sample relation labels + neg_sample_relation_labels]
		labels = [tmp_relation for _ in range(len(pos_examples))] + neg_relations
		ex = {
			't_ex': t_ex,
			'm_ex': m_ex,
			'labels': labels,
			'p_samples': pos_samples,
			'n_samples': neg_samples,
			'direction': direction
		}

		return ex

	def tmp_stance(self, tmp_relation, tm_stance):
		# 0 (entail) + 1 = 1 % 2 = 1
		# 1 (contradict) + 1 = 2 % 2 = 0
		r_mod = (tmp_relation + 1) % 2
		# 1 % 2 = 1 + 1 = 2
		# 2 % 2 = 0 + 1 = 1
		# r_mod is 1 when entail, so flipping (tm_stance + 1) % 2 to get 1 is same as tm_stance
		# r_mod is 0 when contradict, so flipping tm_stance is correct
		return self.flip_tm_stance((tm_stance + r_mod) % 2)

	def flip_tm_stance(self, tm_stance):
		# 0 is no_stance
		# 1 is accept
		# 2 is reject
		# 0 -> 0
		# 1 -> 2
		# 2 -> 1
		# 0 % 2 = 0 + 0 = 0
		# 1 % 2 = 1 + 1 = 2
		# 2 % 2 = 0 + 1 = 1
		tm_flip_stance = (tm_stance % 2) + min(1, tm_stance)
		return tm_flip_stance

	def flip_relation(self, tmp_relation):
		# 0 -> 1
		# 1 -> 0
		return (tmp_relation + 1) % 2

	def _negative_sample(self, m_id, tmp_relation, pm_stance, pos_samples, sample_count):
		possible_permutations = [
			permutation
			for permutation in self.permutations
			if permutation(m_id, tmp_relation, pm_stance) is not None
		]
		p_indices = torch.randint(
			high=len(possible_permutations),
			size=[sample_count]
		).tolist()
		samples = [
			possible_permutations[i](m_id, tmp_relation, pm_stance, pos_samples) for i in p_indices
		]

		return samples

	def flip_polarity(self, m_id, tmp_relation, pm_stance, pos_samples):
		# (flip polarity): a -> a to a -> r
		flip_pm_stance = self.flip_tm_stance(pm_stance)
		m_examples = self.label_examples[m_id][flip_pm_stance]
		if len(m_examples) == 0:
			return None
		s_example = self._sample(
			m_examples,
			m_count=1,
			replacement=True
		)[0]
		return tmp_relation, s_example

	def flip_rel(self, m_id, tmp_relation, pm_stance, pos_samples):
		# (flip rel) a -> a to a \-> a
		flip_tmp_relation = self.flip_relation(tmp_relation)
		s_example = self._sample(
			pos_samples,
			m_count=1,
			replacement=True
		)[0]
		return flip_tmp_relation, s_example

	def zero_polarity(self, m_id, tmp_relation, pm_stance, pos_samples):
		# (zero polarity): a -> a to a -> ns
		# 0 is zero stance polarity
		m_examples = self.label_examples[m_id][0]
		if len(m_examples) == 0:
			return None
		s_example = self._sample(
			m_examples,
			m_count=1,
			replacement=True
		)[0]
		return tmp_relation, s_example

	def _sample(self, m_examples, m_count, replacement=False):
		samples = []
		if m_count <= 0:
			return samples
		if not replacement:
			m_s_indices = torch.randperm(
				n=len(m_examples),
			).tolist()[:m_count]
		else:
			m_s_indices = torch.randint(
				high=len(m_examples),
				size=[m_count]
			).tolist()
		for s_idx in m_s_indices:
			samples.append(m_examples[s_idx])
		return samples

	def _sample_direction(self):
		r = torch.rand(
			size=(1,),
		).tolist()[0]
		if r < 0.5:
			return 0
		else:
			return 1

	def _sample_relation(self):
		r = torch.rand(
			size=(1,),
		).tolist()[0]
		if r < 0.5:
			return 0
		else:
			return 1


class KbiMisinfoStanceDataModule(BaseDataModule):
	def __init__(
			self,
			train_misinfo_path: str = None,
			val_misinfo_path: str = None,
			test_misinfo_path: str = None,
			predict_misinfo_path: str = None,
			pos_samples: int = 1,
			neg_samples: int = 1,
			num_relations: int = 2,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)

		self.train_misinfo_path = train_misinfo_path
		self.val_misinfo_path = val_misinfo_path
		self.test_misinfo_path = test_misinfo_path
		self.predict_misinfo_path = predict_misinfo_path
		self.pos_samples = pos_samples
		self.neg_samples = neg_samples
		self.num_relations = num_relations

		if self.train_path is not None and self.train_misinfo_path is not None:
			self.train_dataset = KbiMisinfoStanceDataset(
				pos_samples=self.pos_samples,
				neg_samples=self.neg_samples,
				tokenizer=self.tokenizer,
				data_path=self.train_path,
				misinfo_path=train_misinfo_path
			)
		if self.val_path is not None and self.val_misinfo_path is not None:
			self.val_dataset = KbiMisinfoStanceDataset(
				pos_samples=self.pos_samples,
				neg_samples=self.neg_samples,
				tokenizer=self.tokenizer,
				data_path=self.val_path,
				misinfo_path=val_misinfo_path
			)
		if self.test_path is not None and self.test_misinfo_path is not None:
			self.test_dataset = KbiMisinfoStanceDataset(
				tokenizer=self.tokenizer,
				data_path=self.test_path,
				misinfo_path=test_misinfo_path
			)
		if self.predict_path is not None and self.predict_misinfo_path is not None:
			self.predict_dataset = KbiMisinfoStanceDataset(
				tokenizer=self.tokenizer,
				data_path=self.predict_path,
				misinfo_path=predict_misinfo_path
			)

	def create_collator(self):
		return KbiBatchCollator(
			num_relations=self.num_relations,
			max_seq_len=self.max_seq_len,
			use_tpus=self.use_tpus,
		)
