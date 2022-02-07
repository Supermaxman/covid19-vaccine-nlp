
from collections import defaultdict
from typing import Optional

import torch
import numpy as np

from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds import ThresholdModule, MultiClassThresholdModule
from pytorch_gleam.modeling.metrics import Metric
from pytorch_gleam.inference import ConsistencyScoring


# noinspection PyAbstractClass
class NliTextLanguageModel(BaseLanguageModel):
	def __init__(
			self,
			infer: ConsistencyScoring,
			threshold: ThresholdModule,
			metric: Metric,
			m_metric: Metric,
			num_classes: int = 3,
			num_val_seeds: int = 1,
			num_threshold_steps: int = 100,
			update_threshold: bool = True,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.num_classes = num_classes
		self.infer = infer
		self.num_val_seeds = num_val_seeds
		self.threshold = threshold
		self.m_metric = m_metric
		self.num_threshold_steps = num_threshold_steps
		self.update_threshold = update_threshold

		self.cls_layer = torch.nn.Linear(
			in_features=self.hidden_size,
			out_features=self.num_classes
		)
		self.criterion = torch.nn.CrossEntropyLoss(
			reduction='none'
		)
		self.score_func = torch.nn.Softmax(
			dim=-1
		)
		self.f_dropout = torch.nn.Dropout(
			p=self.hidden_dropout_prob
		)

		self.metric = metric

	def setup(self, stage: Optional[str] = None):
		super().setup(stage)
		if stage == 'fit':
			data_loader = self.train_dataloader()
		elif stage == 'test':
			data_loader = self.test_dataloader()[0]
		elif stage == 'val':
			data_loader = self.val_dataloader()[0]
		elif stage == 'predict':
			data_loader = self.predict_dataloader()
		else:
			raise ValueError(f'Unknown stage: {stage}')
		misinfo = data_loader.dataset.misinfo
		for m_id, m in misinfo.items():
			if m_id not in self.threshold:
				self.threshold[m_id] = MultiClassThresholdModule()

	def infer_m_scores(self, adj_list, stage_labels, stage):
		# always use stage 0 (val) for seeds
		seed_labels = stage_labels[0]
		seed_examples = [
			(ex_id, label) for (ex_id, label)
			in seed_labels.items()
			if label != 0
		]
		# if the stage is val then we have no test set, so pick
		# some number of seed examples from val and test on remaining val
		if stage == 'val':
			seed_examples = seed_examples[:self.num_val_seeds]
			# make sure adj list only has val labeled data
			adj_list = [
				(u_id, v_id, uv_scores) for (u_id, v_id, uv_scores) in adj_list
				if u_id in seed_labels and v_id in seed_labels
			]
		seed_examples = {
			ex_id: label for (ex_id, label)
			in seed_examples
		}
		if len(adj_list) == 0:
			node_scores = np.zeros([len(seed_labels), 3], dtype=np.float32)
			node_idx_map = {node: idx for (idx, node) in enumerate(seed_labels)}
		else:
			node_scores, node_idx_map = self.infer(adj_list, seed_examples)
		if stage == 'test':
			eval_labels = stage_labels[1]
		else:
			eval_labels = stage_labels[0]
		scores = []
		# make sure we pack example scores in proper order
		for ex_id in eval_labels:
			ex_idx = node_idx_map[ex_id]
			ex_scores = torch.tensor(node_scores[ex_idx], dtype=torch.float32)
			scores.append(ex_scores)
		scores = torch.stack(scores, dim=0)
		return scores

	def eval_epoch_end(self, outputs, stage):
		triplet_eval_outputs, infer_eval_outputs = outputs
		loss = torch.cat([x['loss'] for x in triplet_eval_outputs], dim=0).mean()
		self.log(f'{stage}_loss', loss)

		self.threshold.cpu()
		# stage 0 is validation
		# stage 1 is test
		m_adj_lists, m_stage_labels = build_adj_list(infer_eval_outputs)

		for m_id in m_stage_labels:
			if m_id not in self.threshold:
				self.threshold[m_id] = MultiClassThresholdModule()

		m_s_ids = []
		m_s_m_ids = []
		m_s_labels = []
		m_s_preds = []
		for m_id, stage_labels in m_stage_labels.items():
			m_adj_list = m_adj_lists[m_id]
			m_threshold = self.threshold[m_id]
			if len(m_adj_list) == 0:
				continue
			m_ex_ids = []
			m_ex_m_ids = []
			m_ex_labels = []
			if stage != 'val':
				eval_labels = stage_labels[1]
			else:
				eval_labels = stage_labels[0]
			for ex_id, label in eval_labels.items():
				m_ex_labels.append(label)
				m_ex_ids.append(ex_id)
				m_ex_m_ids.append(m_id)
			m_ex_labels = torch.tensor(m_ex_labels, dtype=torch.long)
			m_ex_scores = self.infer_m_scores(m_adj_list, stage_labels, stage)
			if self.update_threshold:
				m_min_score = torch.min(m_ex_scores).item()
				m_max_score = torch.max(m_ex_scores).item()
				# check 100 values between min and max
				if m_min_score == m_max_score:
					m_max_score += 1
				m_delta = (m_max_score - m_min_score) / self.num_threshold_steps
				max_threshold, max_metrics = self.m_metric.best(
					m_ex_labels,
					m_ex_scores,
					m_threshold,
					threshold_min=m_min_score,
					threshold_max=m_max_score,
					threshold_delta=m_delta,
				)
				m_threshold.update_thresholds(max_threshold)

			m_ex_preds = m_threshold(m_ex_scores)
			m_f1, m_p, m_r, m_cls_f1, m_cls_p, m_cls_r, m_cls_indices = self.m_metric(
				m_ex_labels,
				m_ex_preds
			)
			self.log(f'{stage}_{m_id}_micro_f1', m_f1)
			self.log(f'{stage}_{m_id}_micro_p', m_p)
			self.log(f'{stage}_{m_id}_micro_r', m_r)
			self.log(f'{stage}_{m_id}_threshold', m_threshold.thresholds.item())
			for cls_index, c_f1, c_p, c_r in zip(m_cls_indices, m_cls_f1, m_cls_p, m_cls_r):
				self.log(f'{stage}_{m_id}_{cls_index}_f1', c_f1)
				self.log(f'{stage}_{m_id}_{cls_index}_p', c_p)
				self.log(f'{stage}_{m_id}_{cls_index}_r', c_r)
			m_s_ids.extend(m_ex_ids)
			m_s_m_ids.extend(m_ex_m_ids)
			m_s_labels.append(m_ex_labels)
			m_s_preds.append(m_ex_preds)

		m_s_labels = torch.cat(m_s_labels, dim=0)
		m_s_preds = torch.cat(m_s_preds, dim=0)
		f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(
			m_s_labels,
			m_s_preds
		)
		micro_f1, micro_p, micro_r, _, _, _, _ = self.m_metric(
			m_s_labels,
			m_s_preds
		)
		self.log(f'{stage}_loss', loss)
		self.log(f'{stage}_f1', f1)
		self.log(f'{stage}_p', p)
		self.log(f'{stage}_r', r)
		self.log(f'{stage}_micro_f1', micro_f1)
		self.log(f'{stage}_micro_p', micro_p)
		self.log(f'{stage}_micro_r', micro_r)
		for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
			self.log(f'{stage}_{cls_index}_f1', c_f1)
			self.log(f'{stage}_{cls_index}_p', c_p)
			self.log(f'{stage}_{cls_index}_r', c_r)

		self.threshold.to(self.device)

	def eval_step(self, batch, batch_idx, dataloader_idx=None):
		if dataloader_idx is None or dataloader_idx == 0:
			logits, scores = self(batch)
			loss = self.loss(logits, batch['relations'])
			result = {
				'loss': loss,
			}
		else:
			result = self.predict_step(batch, batch_idx, dataloader_idx)

		return result

	def forward(self, batch):
		input_ids = batch['input_ids']
		attention_mask = batch['attention_mask']
		if 'token_type_ids' in batch:
			token_type_ids = batch['token_type_ids']
		else:
			token_type_ids = None
		# [bsize, seq_len, hidden_size]
		contextualized_embeddings = self.lm_step(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		# [bsize, hidden_size]
		lm_output = contextualized_embeddings[:, 0]
		lm_output = self.f_dropout(lm_output)
		logits = self.cls_layer(lm_output)
		scores = self.score_func(logits)
		return logits, scores

	def loss(self, logits, labels):
		loss = self.criterion(
			logits,
			labels
		)

		return loss

	def training_step(self, batch, batch_idx):
		logits, scores = self(batch)
		loss = self.loss(logits, batch['relations'])

		loss = loss.mean()
		self.log('train_loss', loss)
		result = {
			'loss': loss
		}
		return result

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		logits, scores = self(batch)

		results = {
			# [bsize]
			'ids': batch['ids'],
			# [bsize]
			'm_ids': batch['m_ids'],
			# [bsize]
			'p_ids': batch['p_ids'],
			# [bsize, 2]
			'labels': batch['labels'],
			# [bsize, 2]
			'stages': batch['stages'],
			# [bsize, num_relations]
			# only keep energies of entail and contradict relations, drop no relation
			'energies': -logits[:, [0, 1]]
		}
		return results


def flatten(l):
	return [item for sublist in l for item in sublist]


def build_adj_list(outputs):
	# [count]
	t_ids = flatten([x['ids'] for x in outputs])
	# [count]
	m_ids = flatten([x['m_ids'] for x in outputs])
	# [count]
	p_ids = flatten([x['p_ids'] for x in outputs])
	# [count, 2]
	labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu()
	stages = torch.cat([x['stages'] for x in outputs], dim=0).cpu()
	# [count]
	t_label = labels[:, 0]
	# [count]
	t_stage = stages[:, 0]
	# [count]
	p_labels = labels[:, 1]
	# [count]
	p_stage = stages[:, 1]

	# [count, num_relations]
	t_energies = torch.cat([x['energies'] for x in outputs], dim=0).cpu()

	m_adj_list = defaultdict(list)
	m_labels = defaultdict(lambda: defaultdict(dict))
	for ex_idx in range(len(t_ids)):
		ex_t_id = t_ids[ex_idx]
		ex_m_id = m_ids[ex_idx]
		ex_p_id = p_ids[ex_idx]
		ex_t_label = t_label[ex_idx]
		ex_t_stage = int(t_stage[ex_idx])
		ex_p_label = p_labels[ex_idx]
		ex_p_stage = int(p_stage[ex_idx])
		ex_tmp_energy = t_energies[ex_idx]
		m_labels[ex_m_id][ex_t_stage][ex_t_id] = ex_t_label
		m_adj_list[ex_m_id].append((ex_t_id, ex_p_id, ex_tmp_energy))
		m_labels[ex_m_id][ex_p_stage][ex_p_id] = ex_p_label

	return m_adj_list, m_labels


def build_stage_labels(m_labels):
	m_s_labels = defaultdict(list)
	m_m_ids = defaultdict(list)
	m_t_ids = defaultdict(list)
	for m_id, m_t_labels in m_labels.items():
		for stage_idx, stage_labels in m_t_labels.items():
			for m_t_id, m_t_label in stage_labels.items():
				m_t_ids[stage_idx].append(m_t_id)
				m_m_ids[stage_idx].append(m_id)
				m_s_labels[stage_idx].append(m_t_label)
	return m_s_labels, m_m_ids, m_t_ids
