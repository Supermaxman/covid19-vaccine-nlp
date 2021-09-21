
from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_gleam.data.datasets import KbiMisinfoStanceDataModule
from pytorch_gleam.modeling.models import KbiLanguageModel
from pytorch_gleam.callbacks import FitCheckpointCallback


if __name__ == '__main__':
	cli = LightningCLI(
		KbiLanguageModel,
		KbiMisinfoStanceDataModule,
		run=False,
		trainer_defaults={
			'checkpoint_callback': False,
			'callbacks': [
				FitCheckpointCallback()
			]
		}
	)
	cli.trainer.test(
		cli.model,
		datamodule=cli.datamodule
	)

