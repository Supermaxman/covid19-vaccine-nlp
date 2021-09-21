
from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_gleam.data.datasets import MultiClassMisinfoStanceDataModule
from pytorch_gleam.modeling.models import MultiClassLanguageModel
from pytorch_gleam.callbacks import FitCheckpointCallback


if __name__ == '__main__':
	cli = LightningCLI(
		MultiClassLanguageModel,
		MultiClassMisinfoStanceDataModule,
		run=False,
		trainer_defaults={
			'checkpoint_callback': False,
			'callbacks': [
				FitCheckpointCallback()
			]
		}
	)
	cli.trainer.validate(
		cli.model,
		datamodule=cli.datamodule
	)

