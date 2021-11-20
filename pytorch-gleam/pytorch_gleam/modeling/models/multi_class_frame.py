
from typing import Dict, List

import torch

from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds import ThresholdModule
from pytorch_gleam.modeling.metrics import Metric
from pytorch_gleam.modeling.layers.gcn import GraphAttention
from pytorch_gleam.modeling.layers.hopfield import HopfieldPooling


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
			graphs: List[str],
			gcn_size: int,
			gcn_depth: int,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.graphs = graphs
		self.gcn_size = gcn_size
		self.gcn_depth = gcn_depth

		if self.hidden_size != self.gcn_size:
			self.gcn_projs = torch.nn.ModuleDict(
				{
					f'{graph}_proj': torch.nn.Linear(
						self.hidden_size, self.gcn_size) for graph in self.graphs
				}
			)
		else:
			self.gcn_projs = None
		self.gcn_hidden_size = len(self.graphs) * gcn_size

		self.gcns = torch.nn.ModuleDict()
		for graph_name in self.graphs:
			for d in range(self.gcn_depth):
				layer_name = f'{graph_name}_{d}_gcn'
				# first layer takes bert reduced output,
				# further layers take previous graph outputs
				in_features = gcn_size if d == 0 else self.gcn_hidden_size
				out_features = gcn_size
				self.gcns[layer_name] = GraphAttention(
					in_features=in_features,
					out_features=out_features,
					dropout=self.hidden_dropout_prob,
					alpha=0.2,
					concat=True,
				)

		self.cls_layer = torch.nn.Linear(
			in_features=self.gcn_hidden_size,
			out_features=self.num_classes
		)

	def gcn_pool(self, graph_outputs, graph_mask, batch):
		# [bsize, seq_len] -> [bsize] -> [bsize, 1]
		counts = graph_mask.float().sum(dim=-1).unsqueeze(dim=-1)
		# [bsize, seq_len, hidden_size] -> [bsize, hidden_size] / [bsize, 1] -> [bsize, hidden_size]
		graph_outputs_pooled = graph_outputs.sum(dim=-2) / counts
		return graph_outputs_pooled

	def gcn_forward(self, node_embeddings, batch):
		# [bsize, seq_len, hidden_size]
		graph_inputs = [node_embeddings]
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
		return graph_outputs

	def forward(self, batch):
		# [bsize, seq_len, hidden_size]
		contextualized_embeddings = self.lm_step(
			batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None
		)
		graph_outputs = self.gcn_forward(
			contextualized_embeddings,
			batch
		)
		classifier_inputs = self.gcn_pool(
			graph_outputs,
			graph_mask=batch['attention_mask'],
			batch=batch
		)
		classifier_inputs = self.f_dropout(classifier_inputs)
		logits = self.cls_layer(classifier_inputs)
		return logits


# noinspection PyAbstractClass
class MultiClassFrameGraphMoralityLanguageModel(MultiClassFrameGraphLanguageModel):

	def __init__(
			self,
			morality_map: Dict[str, int],
			hopfield_update_steps_max: int = 2,
			hopfield_dropout: float = 0.0,
			hopfield_num_heads: int = 1,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.morality_map = morality_map
		self.num_moralities = len(self.morality_map)

		self.f_morality_pooler = HopfieldPooling(
			input_size=self.gcn_hidden_size,
			quantity=self.num_moralities,
			update_steps_max=hopfield_update_steps_max,
			dropout=hopfield_dropout,
			num_heads=hopfield_num_heads
		)
		self.ex_morality_pooler = HopfieldPooling(
			input_size=self.gcn_hidden_size,
			quantity=self.num_moralities,
			update_steps_max=hopfield_update_steps_max,
			dropout=hopfield_dropout,
			num_heads=hopfield_num_heads
		)
		self.f_ex_pooler = torch.nn.Linear(
			in_features=self.gcn_hidden_size * self.num_moralities,
			out_features=self.gcn_hidden_size
		)
		self.f_ex_activation = torch.nn.ReLU()

	def gcn_pool(self, graph_outputs, graph_mask, batch):
		# [bsize, num_moralities, 1]
		ex_morality_mask = batch['ex_morality'].float().unsqueeze(dim=-1)
		# [bsize, num_moralities, 1]
		f_morality_mask = batch['f_morality'].float().unsqueeze(dim=-1)
		ex_seq_mask = batch['ex_seq_mask']
		f_seq_mask = batch['f_seq_mask']
		# [bsize, num_moralities, hidden_size]
		f_pool = self.f_morality_pooler(
			graph_outputs,
			# masks used are inverted, aka ignored values should be True
			stored_pattern_padding_mask=~ex_seq_mask.bool()
		)
		# [bsize, num_moralities, hidden_size]
		ex_pool = self.ex_morality_pooler(
			graph_outputs,
			# masks used are inverted, aka ignored values should be True
			stored_pattern_padding_mask=~f_seq_mask.bool()
		)
		print(f_pool.shape)
		print(ex_pool.shape)
		# [bsize, num_moralities, hidden_size]
		f_pool = f_pool * f_morality_mask
		# [bsize, num_moralities, hidden_size]
		ex_pool = ex_pool * ex_morality_mask

		# [bsize, num_moralities, hidden_size]
		f_ex_pool = (f_pool + ex_pool) / 2.0
		# [bsize, num_moralities * hidden_size]
		f_ex_emb = f_ex_pool.view(-1, self.num_moralities * self.gcn_hidden_size)
		# [bsize, hidden_size]
		graph_outputs_pooled = self.f_ex_activation(self.f_ex_pooler(f_ex_emb))

		return graph_outputs_pooled
