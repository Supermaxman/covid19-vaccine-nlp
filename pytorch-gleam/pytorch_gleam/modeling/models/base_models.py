from abc import abstractmethod, ABC
from typing import Optional, Union, Callable

import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup


class BaseLanguageModel(pl.LightningModule, ABC):
	lm: Union[Callable]

	def __init__(
			self,
			pre_model_name: str,
			learning_rate: float = 5e-5,
			weight_decay: float = 0.0,
			lr_warm_up: float = 0.1,
			load_pre_model: bool = True,
			torch_cache_dir: str = None
	):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warm_up = lr_warm_up
		# assigned later when training starts
		self.train_steps = 0
		self.torch_cache_dir = torch_cache_dir

		if load_pre_model:
			self.lm = AutoModel.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
		else:
			config = AutoConfig.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
			self.lm = AutoModel.from_config(config)
		# noinspection PyUnresolvedReferences
		self.hidden_size = self.lm.config.hidden_size
		# noinspection PyUnresolvedReferences
		self.hidden_dropout_prob = self.lm.config.hidden_dropout_prob

	def lm_step(self, input_ids, attention_mask, token_type_ids=None):
		if token_type_ids is not None:
			outputs = self.lm(
				input_ids=input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids
			)
		else:
			outputs = self.lm(
				input_ids=input_ids,
				attention_mask=attention_mask,
			)
		contextualized_embeddings = outputs[0]
		return contextualized_embeddings

	def setup(self, stage: Optional[str] = None):
		if stage == 'fit':
			total_devices = self.trainer.num_nodes * self.trainer.num_gpus
			train_batches = len(self.train_dataloader()) // total_devices
			# need to figure out how many batches will actually have gradient updates
			train_batches = train_batches // self.trainer.accumulate_grad_batches
			self.train_steps = (self.trainer.max_epochs * train_batches)

	def configure_optimizers(self):
		params = self._get_optimizer_params(self.weight_decay)
		optimizer = AdamW(
			params,
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
			correct_bias=False
		)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=self.lr_warm_up * self.train_steps,
			num_training_steps=self.train_steps
		)
		return [optimizer], [scheduler]

	def _get_optimizer_params(self, weight_decay):
		param_optimizer = list(self.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_params = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
		return optimizer_params

	def validation_step(self, batch, batch_idx, dataloader_idx=None):
		return self.eval_step(batch, batch_idx, dataloader_idx)

	def test_step(self, batch, batch_idx, dataloader_idx=None):
		return self.eval_step(batch, batch_idx, dataloader_idx)

	def validation_epoch_end(self, outputs):
		self.eval_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
		self.eval_epoch_end(outputs, 'test')

	@abstractmethod
	def eval_step(self, batch, batch_idx, dataloader_idx=None):
		pass

	@abstractmethod
	def eval_epoch_end(self, outputs, stage):
		pass
