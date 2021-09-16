
from pytorch_lightning.utilities.cli import LightningCLI

from model_utils import *
from data_utils import *
from checkpoint_utils import FitCheckpointCallback
from prediction_utils import JsonlWriter

if __name__ == '__main__':
	cli = LightningCLI(
		MultiClassLanguageModel,
		MultiClassMisinfoDataModule,
		run=False,
		trainer_defaults={
			'checkpoint_callback': False,
			'callbacks': [
				FitCheckpointCallback(),
				JsonlWriter(
					write_interval='epoch'
				)
			]
		}
	)
	cli.trainer.fit(
		cli.model,
		datamodule=cli.datamodule
	)

