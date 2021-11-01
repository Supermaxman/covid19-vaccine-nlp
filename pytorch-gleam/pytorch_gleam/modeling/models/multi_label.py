
from typing import Dict

import torch

from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds import ThresholdModule, MultiClassThresholdModule
from pytorch_gleam.modeling.metrics import Metric


# noinspection PyAbstractClass
class MultiLabelLanguageModel(BaseLanguageModel):
	def __init__(
			self,
			label_map: Dict[str, int],
			threshold: ThresholdModule,
			metric: Metric,
			m_metric: Metric,
			num_classes: int = 2,
			num_threshold_steps: int = 100,
			update_threshold: bool = True,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.label_map = label_map
		self.num_classes = num_classes
		self.num_labels = len(self.label_map)
		self.threshold = threshold
		self.m_metric = m_metric
		self.num_threshold_steps = num_threshold_steps
		self.update_threshold = update_threshold

		self.cls_layer = torch.nn.Linear(
			in_features=self.hidden_size,
			out_features=self.num_labels * self.num_classes
		)
		self.inv_label_map = {v: k for k, v in self.label_map.items()}

		for cls_name in self.label_map.keys():
			self.threshold[cls_name] = MultiClassThresholdModule()

		self.f_dropout = torch.nn.Dropout(
			p=self.hidden_dropout_prob
		)
		self.metric = metric

		self.criterion = torch.nn.CrossEntropyLoss(
			reduction='none'
		)
		self.score_func = torch.nn.Softmax(
			dim=-1
		)

	def eval_epoch_end(self, outputs, stage):
		loss = torch.cat([x['loss'] for x in outputs], dim=0).mean().cpu()
		self.log(f'{stage}_loss', loss)
		self.threshold.cpu()

		results, labels, preds, t_ids = self.eval_outputs(
			outputs,
			stage,
			self.num_threshold_steps,
			self.update_threshold
		)
		for val_name, val in results.items():
			self.log(val_name, val)

		self.threshold.to(self.device)

	def eval_outputs(self, outputs, stage, num_threshold_steps=100, update_threshold=True):
		results = {}

		t_ids = self.flatten([x['ids'] for x in outputs])
		# [count, num_labels]
		labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu()
		# [count, num_labels, num_classes]
		scores = torch.cat([x['scores'] for x in outputs], dim=0).cpu()

		m_s_preds = []
		m_s_labels = []
		for label_name, label_idx in self.label_map.items():
			m_threshold = self.threshold[label_name]
			m_ex_scores = scores[:, label_idx]
			m_ex_labels = labels[:, label_idx]
			if update_threshold:
				m_min_score = torch.min(m_ex_scores).item()
				m_max_score = torch.max(m_ex_scores).item()
				# check 100 values between min and max
				if m_min_score == m_max_score:
					m_max_score += 1
				m_delta = (m_max_score - m_min_score) / num_threshold_steps
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
			results[f'{stage}_{label_name}_micro_f1'] = m_f1
			results[f'{stage}_{label_name}_micro_p'] = m_p
			results[f'{stage}_{label_name}_micro_r'] = m_r
			results[f'{stage}_{label_name}_threshold'] = m_threshold.thresholds.item()
			for cls_index, c_f1, c_p, c_r in zip(m_cls_indices, m_cls_f1, m_cls_p, m_cls_r):
				results[f'{stage}_{label_name}_{cls_index}_f1'] = c_f1
				results[f'{stage}_{label_name}_{cls_index}_p'] = c_p
				results[f'{stage}_{label_name}_{cls_index}_r'] = c_r
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
		results[f'{stage}_f1'] = f1
		results[f'{stage}_p'] = p
		results[f'{stage}_r'] = r
		results[f'{stage}_micro_f1'] = micro_f1
		results[f'{stage}_micro_p'] = micro_p
		results[f'{stage}_micro_r'] = micro_r
		for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
			results[f'{stage}_{cls_index}_f1'] = c_f1
			results[f'{stage}_{cls_index}_p'] = c_p
			results[f'{stage}_{cls_index}_r'] = c_r

		return results, m_s_labels, m_s_preds, t_ids

	def eval_step(self, batch, batch_idx, dataloader_idx=None):
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
		# [bsize, num_labels, num_classes]
		logits = self.cls_layer(lm_output).view(-1, self.num_labels, self.num_classes)
		return logits

	def loss(self, logits, labels):
		# if logits has [bsize, num_labels, ..., num_classes]
		# then we swap the first num_labels axis with num_classes
		# for the criterion for pytorch
		if len(logits.shape) > 2:
			logits = torch.swapaxes(logits, -1, 1)
		loss = self.criterion(
			logits,
			labels
		)
		return loss

	def training_step(self, batch, batch_idx):
		batch_logits = self(batch)
		batch_labels = batch['labels']
		batch_loss = self.loss(
			batch_logits,
			batch_labels
		)
		loss = batch_loss.mean()
		self.log('train_loss', loss)
		result = {
			'loss': loss
		}
		return result

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		logits = self(batch)
		loss = self.loss(logits, batch['labels'])
		scores = self.score_func(logits)

		results = {
			# [bsize]
			'ids': batch['ids'],
			'labels': batch['labels'],
			'stages': batch['stages'],
			'logits': logits,
			'loss': loss,
			'scores': scores,
		}
		return results

	@staticmethod
	def flatten(l):
		return [item for sublist in l for item in sublist]
