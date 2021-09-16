
from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_gleam.data.datasets.misinfo_stance import MultiClassMisinfoDataModule
from pytorch_gleam.modeling.models.multi_class import MultiClassLanguageModel
from pytorch_gleam.callbacks.prediction import JsonlWriter
from pytorch_gleam.callbacks.checkpoint import FitCheckpointCallback


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
	cli.trainer.predict(
		cli.model,
		datamodule=cli.datamodule
	)

