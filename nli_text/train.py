
from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_gleam.modeling.models import NliTextLanguageModel
from pytorch_gleam.data.datasets import NliTextMisinfoStanceDataModule

if __name__ == '__main__':
	cli = LightningCLI(
		NliTextLanguageModel,
		NliTextMisinfoStanceDataModule,
		run=False,
		subclass_mode_model=True,
		subclass_mode_data=True
	)
	cli.trainer.fit(
		cli.model,
		datamodule=cli.datamodule
	)

