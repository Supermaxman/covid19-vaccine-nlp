
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == '__main__':
	cli = LightningCLI(
		pl.LightningModule,
		pl.LightningDataModule,
		run=False,
		subclass_mode_model=True,
		subclass_mode_data=True
	)
	cli.trainer.fit(
		cli.model,
		datamodule=cli.datamodule
	)

