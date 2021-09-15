
import argparse
from pytorch_lightning.utilities.cli import LightningCLI
import pytorch_lightning as pl

from model_utils import get_model
from data_utils import get_data_module


if __name__ == '__main__':
	# PL_CONFIG=config_path to set from config
	# set default_root_dir=models
	# TODO pass these in args
	model_cls = get_model('mc_lm')
	data_module_cls = get_data_module('mc_misinfo')
	cli = LightningCLI(
		model_cls,
		data_module_cls,
		run=False,
	)
	cli.trainer.fit(cli.model, datamodule=cli.datamodule)
	# cli.trainer.test(ckpt_path="best")
	# predictions = cli.trainer.predict(ckpt_path="best")
