
import os
from jsonargparse import ArgumentParser

from pytorch_lightning.utilities.cli import LightningCLI
import torch

from model_utils import get_model
from data_utils import get_data_module


if __name__ == '__main__':
	parser = ArgumentParser(
		prog='app',
		description='Description for my app.'
	)

	parser.add_argument(
		'--model_type',
		type=str,
		default='mc_lm',
		help='Help for option 1.'
	)
	args = parser.parse_args()
	# PL_CONFIG=config_path to set from config
	# TODO pass these in args
	model_cls = get_model(args.model_type)
	data_module_cls = get_data_module('mc_misinfo')
	cli = LightningCLI(
		model_cls,
		data_module_cls,
		run=False,
	)
	cli.trainer.fit(cli.model, datamodule=cli.datamodule)
	checkpoint_path = os.path.join(cli.trainer.default_root_dir, 'pytorch_model.bin')

	if cli.trainer.should_rank_save_checkpoint:
		cli.trainer.model.to('cpu')
		torch.save(cli.trainer.model.state_dict(), checkpoint_path)
