
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
			'checkpoint_callback': False,
			'callbacks': [
				ModelCheckpoint(
					save_weights_only=True,
					monitor=None
				),
				# EarlyStopping(
				# 	monitor='val_f1',
				# 	min_delta=0.01,
				# 	patience=3,
				# 	mode='max'
				# )
			]
		}
	)
	cli.trainer.fit(cli.model, datamodule=cli.datamodule)
