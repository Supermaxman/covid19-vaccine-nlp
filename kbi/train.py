
import os

from pytorch_lightning.utilities.cli import LightningCLI

from model_utils import *
from data_utils import *


if __name__ == '__main__':
	cli = LightningCLI(
		MultiClassLanguageModel,
		MultiClassMisinfoDataModule,
		run=False,
		trainer_defaults={
			'checkpoint_callback': False
		}
	)
	cli.trainer.fit(cli.model, datamodule=cli.datamodule)
	checkpoint_path = os.path.join(cli.trainer.default_root_dir, 'pytorch_model.bin')
	if cli.trainer.should_rank_save_checkpoint:
		cli.trainer.model.to('cpu')
		torch.save(cli.trainer.model.state_dict(), checkpoint_path)
