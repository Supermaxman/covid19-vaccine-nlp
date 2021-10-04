
import torch

from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.datasets.kbi_misinfo_stance import KbiMisinfoInferStanceDataset, KbiMisinfoStanceDataset
from pytorch_gleam.data.collators import NliTextBatchCollator


class NliTextMisinfoStanceDataset(KbiMisinfoStanceDataset):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __getitem__(self, idx):
		ex = super().__getitem__(idx)
		t_ex = ex['t_ex']
		p_ex = ex['p_samples'][0]
		p_label, _ = ex['relations']
		direction = ex['direction']
		n_ex = ex['n_samples'][0]

		label = self._sample_label()
		# pos sample
		if label == 0:
			# 0 - entail
			# 1 - contradict
			# 2 - no relation
			s_label = p_label
			s_ex = p_ex
		# neg_sample
		else:
			# negative examples have no relation
			s_label = 2
			s_ex = n_ex
		if direction == 0:
			token_data = self.tokenizer(
				t_ex['t_text'],
				s_ex['t_text']
			)
		else:
			token_data = self.tokenizer(
				s_ex['t_text'],
				t_ex['t_text']
			)
		labels = [t_ex['m_label'], s_ex['m_label']]
		stages = [t_ex['stage'], s_ex['stage']]
		ex = {
			't_ex': ex['t_ex'],
			'm_ex': ex['m_ex'],
			'p_ex': s_ex,
			'labels': labels,
			'relations': s_label,
			'stages': stages,
			'input_ids': token_data['input_ids'],
			'attention_mask': token_data['attention_mask'],
		}
		if 'token_type_ids' in token_data:
			ex['token_type_ids'] = token_data['token_type_ids']

		return ex

	def _sample_label(self):
		r = torch.rand(
			size=(1,),
		).tolist()[0]
		if r < 0.5:
			return 0
		else:
			return 1


class NliTextMisinfoInferStanceDataset(KbiMisinfoInferStanceDataset):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		ex = self.examples[idx]
		t_ex = ex['t_ex']
		m_ex = ex['m_ex']
		p_ex = ex['p_ex']
		token_data = self.tokenizer(
			t_ex['t_text'],
			p_ex['t_text']
		)

		labels = [t_ex['m_label'], p_ex['m_label']]
		stages = [t_ex['stage'], p_ex['stage']]
		# unknown relations
		relations = 2
		ex = {
			't_ex': t_ex,
			'm_ex': m_ex,
			'p_ex': p_ex,
			'labels': labels,
			'stages': stages,
			'relations': relations,
			'input_ids': token_data['input_ids'],
			'attention_mask': token_data['attention_mask'],
		}
		if 'token_type_ids' in token_data:
			ex['token_type_ids'] = token_data['token_type_ids']

		return ex


class NliTextMisinfoStanceDataModule(BaseDataModule):
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
		self.pos_samples = 1
		self.neg_samples = 1

		if self.train_path is not None and self.train_misinfo_path is not None:
			self.train_dataset = NliTextMisinfoStanceDataset(
				pos_samples=self.pos_samples,
				neg_samples=self.neg_samples,
				tokenizer=self.tokenizer,
				data_path=self.train_path,
				misinfo_path=self.train_misinfo_path
			)
		if self.val_path is not None and self.val_misinfo_path is not None:
			val_triplet_dataset = NliTextMisinfoStanceDataset(
				pos_samples=self.pos_samples,
				neg_samples=self.neg_samples,
				tokenizer=self.tokenizer,
				data_path=self.val_path,
				misinfo_path=self.val_misinfo_path
			)
			val_infer_dataset = NliTextMisinfoInferStanceDataset(
				pos_samples=1,
				neg_samples=1,
				tokenizer=self.tokenizer,
				data_path=self.val_path,
				misinfo_path=self.val_misinfo_path
			)

			self.val_dataset = [
				val_triplet_dataset,
				val_infer_dataset
			]
		if self.test_path is not None and self.test_misinfo_path is not None:
			test_triplet_dataset = NliTextMisinfoStanceDataset(
				pos_samples=self.pos_samples,
				neg_samples=self.neg_samples,
				tokenizer=self.tokenizer,
				data_path=self.test_path,
				misinfo_path=self.test_misinfo_path
			)
			test_infer_dataset = NliTextMisinfoInferStanceDataset(
				pos_samples=1,
				neg_samples=1,
				tokenizer=self.tokenizer,
				data_path=[self.val_path, self.test_path],
				misinfo_path=self.test_misinfo_path
			)

			self.test_dataset = [
				test_triplet_dataset,
				test_infer_dataset
			]
		if self.predict_path is not None and self.predict_misinfo_path is not None:
			self.predict_dataset = NliTextMisinfoInferStanceDataset(
				pos_samples=1,
				neg_samples=1,
				tokenizer=self.tokenizer,
				data_path=[self.val_path, self.predict_path],
				misinfo_path=self.predict_misinfo_path
			)

	def create_collator(self):
		return NliTextBatchCollator(
			max_seq_len=self.max_seq_len,
			use_tpus=self.use_tpus,
		)
