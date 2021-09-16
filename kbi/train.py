
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from model_utils import *
from data_utils import *


if __name__ == '__main__':
	checkpoint_callback = ModelCheckpoint(
		save_weights_only=True,
		monitor=None,
		save_top_k=0,
		save_last=False
	)
	cli = LightningCLI(
		MultiClassLanguageModel,
		MultiClassMisinfoDataModule,
		run=False,
		trainer_defaults={
			'callbacks': [
				checkpoint_callback
			]
		}
	)
	cli.trainer.fit(cli.model, datamodule=cli.datamodule)
