
import torch

from pytorch_gleam.modeling.models.base_models import BaseLanguageModelForSequenceClassification
from pytorch_gleam.modeling.thresholds.multi_class import MultiClassThresholdModule
from pytorch_gleam.modeling.metrics.multi_class_f1 import F1PRMultiClassMetric


# noinspection PyAbstractClass
class NliMisinfoLanguageModel(BaseLanguageModelForSequenceClassification):
	def __init__(
			self,
			num_classes: int = 3,
			metric: str = 'f1',
			metric_mode: str = 'macro',
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.num_classes = num_classes
		self.criterion = torch.nn.CrossEntropyLoss(
			reduction='none'
		)
		self.score_func = torch.nn.Softmax(
			dim=-1
		)
		self.threshold = MultiClassThresholdModule()
		# TODO select based on metric
		self.metric = F1PRMultiClassMetric(
			num_classes=self.num_classes,
			mode=metric_mode
		)

	def forward(self, batch):
		if 'token_type_ids' in batch:
			token_type_ids = batch['token_type_ids']
		else:
			token_type_ids = None
		logits = self.lm_step(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=token_type_ids,
		)
		return logits

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		batch_logits = self(batch)
		batch_scores = self.score_func(batch_logits)
		batch_preds = self.threshold(batch_scores)
		batch_ids = batch['ids']
		results = {
			'ids': batch_ids,
			'logits': batch_logits,
			'scores': batch_scores,
			'preds': batch_preds
		}
		return results

	def eval_epoch_end(self, outputs, stage):
		loss = torch.cat([x['loss'] for x in outputs], dim=0).mean()
		scores = torch.cat([x['scores'] for x in outputs], dim=0)
		labels = torch.cat([x['labels'] for x in outputs], dim=0)
		scores = scores.cpu()
		labels = labels.cpu()
		self.threshold.cpu()

		if stage == 'val':
			# select max f1 threshold
			max_threshold, max_metrics = self.metric.best(
				labels,
				scores,
				self.threshold
			)
			self.threshold.update_thresholds(max_threshold)
		preds = self.threshold(scores)

		f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(
			labels,
			preds
		)
		self.log(f'{stage}_loss', loss)
		self.log(f'{stage}_f1', f1)
		self.log(f'{stage}_p', p)
		self.log(f'{stage}_r', r)
		for t_idx, threshold in enumerate(self.threshold.thresholds):
			self.log(f'{stage}_threshold_{t_idx}', threshold)
		for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
			self.log(f'{stage}_{cls_index}_f1', c_f1)
			self.log(f'{stage}_{cls_index}_p', c_p)
			self.log(f'{stage}_{cls_index}_r', c_r)

		self.threshold.to(self.device)

	def training_step(self, batch, batch_idx):
		batch_logits = self(batch)
		batch_labels = batch['labels']
		batch_loss = self.criterion(
			batch_logits,
			batch_labels
		)
		loss = batch_loss.mean()
		self.log('train_loss', loss)
		result = {
			'loss': loss
		}
		return result

	def eval_step(self, batch, batch_idx, dataloader_idx=None):
		results = self.predict_step(batch, batch_idx, dataloader_idx)
		loss = self.criterion(
			results['logits'],
			batch['labels']
		)
		results['loss'] = loss
		results['labels'] = batch['labels']
		return results
