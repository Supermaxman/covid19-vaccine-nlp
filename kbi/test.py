
from pytorch_lightning.utilities.cli import LightningCLI

from model_utils import *
from data_utils import *
from checkpoint_utils import FitCheckpointCallback

if __name__ == '__main__':
	cli = LightningCLI(
		MultiClassLanguageModel,
		MultiClassMisinfoDataModule,
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

