
from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_gleam.data.datasets import MultiClassMisinfoStanceDataModule
from pytorch_gleam.modeling.models import NliMisinfoLanguageModel


if __name__ == '__main__':
	cli = LightningCLI(
		NliMisinfoLanguageModel,
		MultiClassMisinfoStanceDataModule,
		run=False,
		subclass_mode_model=True,
		subclass_mode_data=True
	)
	cli.trainer.validate(
		cli.model,
		datamodule=cli.datamodule
	)
	cli.trainer.test(
		cli.model,
		datamodule=cli.datamodule
	)

