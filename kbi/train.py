
from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_gleam.data.datasets import KbiMisinfoStanceDataModule
from pytorch_gleam.modeling.models import KbiLanguageModel
from pytorch_gleam.callbacks import FitCheckpointCallback
from pytorch_gleam.callbacks import CopyConfigCallback


if __name__ == '__main__':
	cli = LightningCLI(
		KbiLanguageModel,
		KbiMisinfoStanceDataModule,
		run=False,
		subclass_mode_model=True,
		subclass_mode_data=True,
		trainer_defaults={
			'checkpoint_callback': False,
			'callbacks': [
				FitCheckpointCallback(),
				CopyConfigCallback()
			]
		}
	)
	cli.trainer.fit(
		cli.model,
		datamodule=cli.datamodule
	)

