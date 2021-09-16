
import os
import torch

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn


class FitCheckpointCallback(Callback):
	def __init__(self):
		super().__init__()

	def _get_checkpoint_path(self, trainer: pl.Trainer):
		checkpoint_path = os.path.join(trainer.default_root_dir, 'pytorch_model.bin')
		return checkpoint_path

	def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		if trainer.should_rank_save_checkpoint:
			checkpoint_path = self._get_checkpoint_path(trainer)
			print(f'Saving checkpoint...')
			pl_module.to('cpu')
			torch.save(pl_module.state_dict(), checkpoint_path)

	def _load_fit_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		print(f'Loading checkpoint...')
		checkpoint_path = self._get_checkpoint_path(trainer)
		pl_module.load_state_dict(torch.load(checkpoint_path))

	def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		if trainer.state.fn != TrainerFn.FITTING:
			self._load_fit_checkpoint(trainer, pl_module)

	def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		if trainer.state.fn != TrainerFn.FITTING:
			self._load_fit_checkpoint(trainer, pl_module)

	def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		if trainer.state.fn != TrainerFn.FITTING:
			self._load_fit_checkpoint(trainer, pl_module)
