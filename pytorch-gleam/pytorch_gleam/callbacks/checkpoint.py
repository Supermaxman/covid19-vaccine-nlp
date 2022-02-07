
import os
import torch

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import rank_zero_only


class FitCheckpointCallback(Callback):
	def __init__(self):
		super().__init__()

	def _get_checkpoint_path(self, trainer: pl.Trainer):
		checkpoint_path = os.path.join(trainer.default_root_dir, 'pytorch_model.bin')
		return checkpoint_path

	@rank_zero_only
	def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		checkpoint_path = self._get_checkpoint_path(trainer)
		print(f'Saving checkpoint...')
		pl_module.to('cpu')
		torch.save(pl_module.state_dict(), checkpoint_path)

	def _load_fit_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		print(f'Loading checkpoint...')
		checkpoint_path = self._get_checkpoint_path(trainer)
		pl_module.load_state_dict(torch.load(checkpoint_path), strict=False)

	def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		if trainer.state.fn != TrainerFn.FITTING:
			self._load_fit_checkpoint(trainer, pl_module)

	def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		if trainer.state.fn != TrainerFn.FITTING:
			self._load_fit_checkpoint(trainer, pl_module)

	def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		if trainer.state.fn != TrainerFn.FITTING:
			self._load_fit_checkpoint(trainer, pl_module)


class PreTrainedCheckpointCallback(Callback):
	def __init__(self, pre_model_path: str, pre_checkpoint_name: str = 'pytorch_model.bin'):
		super().__init__()
		self.pre_model_path = pre_model_path
		self.pre_checkpoint_name = pre_checkpoint_name
		self.checkpoint_path = os.path.join(self.pre_model_path, self.pre_checkpoint_name)

	def _load_pre_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		print(f'Loading checkpoint...')
		pl_module.load_state_dict(torch.load(self.checkpoint_path), strict=False)

	def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		self._load_pre_checkpoint(trainer, pl_module)
