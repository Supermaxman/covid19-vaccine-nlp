
from abc import abstractmethod, ABC
from typing import Optional, Union, Callable, Type

import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup


class BasePreModel(pl.LightningModule, ABC):
	lm: Union[Callable]

	def __init__(
			self,
			pre_model_name: str,
			pre_model_type: Type[Union[AutoModel, AutoModelForSequenceClassification]] = AutoModel,
			learning_rate: float = 5e-5,
			weight_decay: float = 0.0,
			lr_warm_up: float = 0.1,
			load_pre_model: bool = True,
			torch_cache_dir: str = None
	):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.pre_model_type = pre_model_type
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warm_up = lr_warm_up
		# assigned later when training starts
		self.train_steps = 0
		self.torch_cache_dir = torch_cache_dir
		if load_pre_model:
			self.lm = self.pre_model_type.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
		else:
			config = AutoConfig.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
			self.lm = self.pre_model_type.from_config(config)

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

	@abstractmethod
	def lm_step(self, input_ids, attention_mask, token_type_ids=None):
		pass


class BaseLanguageModel(BasePreModel, ABC):
	def __init__(
			self,
			pre_model_name: str,
			pre_model_type: Type[Union[AutoModel, AutoModelForSequenceClassification]] = AutoModel,
			*args,
			**kwargs
	):
		super().__init__(pre_model_name, pre_model_type, *args, **kwargs)

		# TODO check for these, not all models may have them
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


class BaseLanguageModelForSequenceClassification(BasePreModel, ABC):
	def __init__(
			self,
			pre_model_name: str,
			pre_model_type: Type[Union[AutoModel, AutoModelForSequenceClassification]] = AutoModelForSequenceClassification,
			*args,
			**kwargs
	):
		super().__init__(pre_model_name, pre_model_type, *args, **kwargs)
		# TODO check for these, not all models may have them
		# noinspection PyUnresolvedReferences
		# self.id2label = self.lm.config.id2label
		# noinspection PyUnresolvedReferences
		# self.label2id = self.lm.config.label2id
		# 0 - contradiction
		# 1 - neutral
		# 2 - entailment
		# want
		# 0 - neutral
		# 1 - entailment
		# 2 - contradiction
		# map
		# 1 -> 0
		# 2 -> 1
		# 0 -> 2
		# TODO build automatically
		# self.label_list = [1, 2, 0]

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

		logits = outputs[0]
		# re-arrange logits
		# logits = logits[:, self.label_list]
		return logits
