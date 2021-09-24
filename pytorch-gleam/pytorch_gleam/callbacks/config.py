
import os
import shutil

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


class CopyConfigCallback(Callback):
	def __init__(self):
		super().__init__()

	def _get_config_path(self, trainer: pl.Trainer):
		# get lightning_logs/version_0/config.yaml
		config_path = os.path.join(trainer.logger.save_dir, 'config.yaml')
		return config_path

	def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
		if trainer.should_rank_save_checkpoint:
			config_path = self._get_config_path(trainer)
			print(f'Saving config...')
			root_config_path = os.path.join(trainer.default_root_dir, 'config.yaml')
			print(config_path)
			print(root_config_path)
			shutil.copyfile(config_path, root_config_path)
