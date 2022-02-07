
import os
import json

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class JsonSaveResultsCallback(Callback):
	def __init__(self):
		super().__init__()

	def _get_results_path(self, trainer: pl.Trainer):
		results_dir = os.path.join(trainer.default_root_dir, 'results')
		if not os.path.exists(results_dir):
			os.mkdir(results_dir)

		results_file = os.path.join(results_dir, 'results.json')
		return results_file

	@rank_zero_only
	def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		results_path = self._get_results_path(trainer)
		results = trainer.accelerator.results
		print(f'Saving results...')
		with open(results_path, 'w') as f:
			json.dump(results, f, indent=2)
