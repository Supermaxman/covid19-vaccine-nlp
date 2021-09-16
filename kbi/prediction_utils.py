
import os
from typing import Any, List

from pytorch_lightning.callbacks import BasePredictionWriter
import pytorch_lightning as pl


class JsonlWriter(BasePredictionWriter):
	def __init__(self, write_interval: str):
		super().__init__(write_interval)

	def write_on_batch_end(
			self,
			trainer: pl.Trainer,
			pl_module: pl.LightningModule,
			prediction: Any,
			batch_indices: List[int],
			batch: Any,
			batch_idx: int,
			dataloader_idx: int
	):
		predictions_dir = os.path.join(trainer.default_root_dir, 'predictions')
		if not os.path.exists(predictions_dir):
			os.mkdir(predictions_dir)

	def write_on_epoch_end(
			self,
			trainer: pl.Trainer,
			pl_module: pl.LightningModule,
			predictions: List[Any],
			batch_indices: List[Any]
	):
		predictions_dir = os.path.join(trainer.default_root_dir, 'predictions')
		if not os.path.exists(predictions_dir):
			os.mkdir(predictions_dir)
