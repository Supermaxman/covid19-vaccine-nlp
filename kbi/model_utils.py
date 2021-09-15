from typing import Type

import torch
import pytorch_lightning as pl

from lm_utils import BaseLanguageModel


def get_model(model_type: str) -> Type[pl.LightningModule]:
	model_map = {
		'mc_lm': MultiClassLanguageModel
	}
	model_type = model_type.lower()
	if model_type not in model_map:
		raise ValueError(f'Unknown model_type: {model_type}')
	return model_map[model_type]


class MultiClassLanguageModel(BaseLanguageModel):
	def __init__(self, num_classes: int = 3, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.num_classes = num_classes
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

	def forward(self, batch):
		contextualized_embeddings = self.lm_step(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		cls_embedding = contextualized_embeddings[:, 0]
		logits = self.cls_layer(cls_embedding)
		return logits

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

	def validation_step(self, batch, batch_idx, dataloader_idx=None):
		return self._eval_step(batch, batch_idx, dataloader_idx=None)

	def test_step(self, batch, batch_idx, dataloader_idx=None):
		return self._eval_step(batch, batch_idx, dataloader_idx=None)

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		batch_logits = self(batch)
		batch_ids = batch['ids']
		assert batch_logits.shape[0] == len(batch_ids)
		results = {
			'ids': batch_ids,
			'logits': batch_logits
		}
		return results

	def validation_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'test')

	def _eval_step(self, batch, batch_idx, dataloader_idx=None):
		results = self.predict_step(batch, batch_idx, dataloader_idx)
		los
		s = self.criterion(
			results['logits'],
			batch['labels']
		)
		results['loss'] = loss
		results['labels'] = batch['labels']
		return results

	def _eval_epoch_end(self, outputs, stage):
		loss = torch.cat([x['loss'] for x in outputs], dim=0).mean()
		logits = torch.cat([x['logits'] for x in outputs], dim=0)
		labels = torch.cat([x['labels'] for x in outputs], dim=0)
		# TODO other metrics
		self.log(f'{stage}_loss', loss)
