
from typing import Dict, List

import torch

from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds import ThresholdModule
from pytorch_gleam.modeling.metrics import Metric
from pytorch_gleam.modeling.layers.gcn import GraphAttention


# noinspection PyAbstractClass
class MultiClassFrameLanguageModel(BaseLanguageModel):
	def __init__(
			self,
			label_map: Dict[str, int],
			threshold: ThresholdModule,
			metric: Metric,
			num_threshold_steps: int = 100,
			update_threshold: bool = True,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.label_map = label_map
		self.num_classes = len(label_map)
		self.threshold = threshold
		self.num_threshold_steps = num_threshold_steps
		self.update_threshold = update_threshold

		self.cls_layer = torch.nn.Linear(
			in_features=self.hidden_size,
			out_features=self.num_classes
		)
		self.inv_label_map = {v: k for k, v in self.label_map.items()}

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
		# [count]
		labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu()
		# [count, num_classes]
		scores = torch.cat([x['scores'] for x in outputs], dim=0).cpu()

		self.threshold.cpu()
		if update_threshold:
			m_min_score = torch.min(scores).item()
			m_max_score = torch.max(scores).item()
			# check 100 values between min and max
			if m_min_score == m_max_score:
				m_max_score += 1
			m_delta = (m_max_score - m_min_score) / num_threshold_steps
			max_threshold, max_metrics = self.metric.best(
				labels,
				scores,
				self.threshold,
				threshold_min=m_min_score,
				threshold_max=m_max_score,
				threshold_delta=m_delta,
			)
			self.threshold.update_thresholds(max_threshold)
		preds = self.threshold(scores)

		f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(
			labels,
			preds
		)

		results[f'{stage}_f1'] = f1
		results[f'{stage}_p'] = p
		results[f'{stage}_r'] = r

		for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
			label_name = self.inv_label_map[cls_index]
			results[f'{stage}_{label_name}_f1'] = c_f1
			results[f'{stage}_{label_name}_p'] = c_p
			results[f'{stage}_{label_name}_r'] = c_r

		return results, labels, preds, t_ids

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
		# [bsize, num_classes]
		logits = self.cls_layer(lm_output)
		return logits

	def loss(self, logits, labels):
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
			'logits': logits,
			'loss': loss,
			'scores': scores,
		}
		return results

	@staticmethod
	def flatten(multi_list):
		return [item for sub_list in multi_list for item in sub_list]


# noinspection PyAbstractClass
class MultiClassFrameGraphLanguageModel(MultiClassFrameLanguageModel):
	def __init__(
			self,
			graphs: list[str],
			gcn_size: int,
			gcn_depth: int,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.graphs = graphs
		self.gcn_size = gcn_size
		self.gcn_depth = gcn_depth

		if self.config.hidden_size != self.gcn_size:
			self.gcn_projs = torch.nn.ModuleDict(
				{
					f'{graph}_proj': torch.nn.Linear(
						self.config.hidden_size, self.gcn_size) for graph in self.graphs
				}
			)
		else:
			self.gcn_projs = None

		self.gcns = torch.nn.ModuleDict()
		for graph_name in self.graphs:
			for d in range(self.gcn_depth):
				layer_name = f'{graph_name}_{d}_gcn'
				# first layer takes bert reduced output,
				# further layers take previous graph outputs
				in_features = gcn_size if d == 0 else gcn_size * len(self.graphs)
				out_features = gcn_size
				self.gcns[layer_name] = GraphAttention(
					in_features=in_features,
					out_features=out_features,
					dropout=self.config.hidden_dropout_prob,
					alpha=0.2,
					concat=True,
				)

		self.cls_layer = torch.nn.Linear(
			in_features=len(self.graphs) * gcn_size,
			out_features=self.num_classes
		)

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
		# [bsize, seq_len, hidden_size]
		graph_inputs = [contextualized_embeddings]
		for d in range(self.gcn_depth):
			graph_emb_inputs = torch.cat(graph_inputs, dim=-1)
			graph_outputs = []
			for graph_name in self.graphs:
				gcn_edges = batch[f'{graph_name}_edges']
				if d == 0 and self.gcn_projs is not None:
					gcn_inputs = self.gcn_projs[f'{graph_name}_proj'](graph_emb_inputs)
				else:
					gcn_inputs = graph_emb_inputs
				gcn_outputs = self.gcns[f'{graph_name}_{d}_gcn'](gcn_inputs, gcn_edges)
				graph_outputs.append(gcn_outputs)
			graph_inputs = graph_outputs
		graph_outputs = torch.cat(graph_inputs, dim=-1)
		# [bsize, seq_len] -> [bsize] -> [bsize, 1]
		counts = attention_mask.float().sum(dim=-1).unsqueeze(dim=-1)
		# [bsize, seq_len, hidden_size] -> [bsize, hidden_size] / [bsize, 1] -> [bsize, hidden_size]
		graph_outputs_pooled = graph_outputs.sum(dim=-2) / counts
		classifier_inputs = graph_outputs_pooled
		classifier_inputs = self.f_dropout(classifier_inputs)
		logits = self.cls_layer(classifier_inputs)
		return logits
