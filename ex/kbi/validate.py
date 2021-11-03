
from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_gleam.data.datasets import KbiMisinfoStanceDataModule
from pytorch_gleam.modeling.models import KbiLanguageModel

if __name__ == '__main__':
	cli = LightningCLI(
		KbiLanguageModel,
		KbiMisinfoStanceDataModule,
		run=False,
		subclass_mode_model=True,
		subclass_mode_data=True
	)
	cli.trainer.validate(
		cli.model,
		datamodule=cli.datamodule
	)

