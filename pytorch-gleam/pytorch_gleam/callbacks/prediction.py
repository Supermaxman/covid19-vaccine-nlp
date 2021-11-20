import json
import os
import torch
from collections import defaultdict
from typing import Any, List

from pytorch_lightning.callbacks import BasePredictionWriter
import pytorch_lightning as pl


class JsonlWriter(BasePredictionWriter):
	def __init__(self, write_interval: str = 'batch'):
		super().__init__(write_interval)

	def _write(self, trainer, prediction):
		predictions_dir = os.path.join(trainer.default_root_dir, 'predictions')
		if not os.path.exists(predictions_dir):
			os.mkdir(predictions_dir)
		pred_file = os.path.join(predictions_dir, 'predictions.jsonl')
		rows = defaultdict(dict)
		for key, values in prediction.items():
			for ex_idx, ex_value in enumerate(values):
				if isinstance(ex_value, torch.Tensor):
					ex_value = ex_value.tolist()
				rows[ex_idx][key] = ex_value
		rows = rows.values()
		with open(pred_file, 'a') as f:
			for row in rows:
				f.write(json.dumps(row) + '\n')

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
		# TODO get node id
		self._write(trainer, prediction)

	def write_on_epoch_end(
			self,
			trainer: pl.Trainer,
			pl_module: pl.LightningModule,
			predictions: List[Any],
			batch_indices: List[Any]
	):
		# TODO get node id
		for prediction in predictions:
			self._write(trainer, prediction)
