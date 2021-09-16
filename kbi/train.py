
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from model_utils import *
from data_utils import *


if __name__ == '__main__':
	cli = LightningCLI(
		MultiClassLanguageModel,
		MultiClassMisinfoDataModule,
		run=False,
		trainer_defaults={
			'callbacks': [
				ModelCheckpoint(
					save_weights_only=True,
					monitor=None
				),
			]
		}
	)
	cli.trainer.fit(cli.model, datamodule=cli.datamodule)
