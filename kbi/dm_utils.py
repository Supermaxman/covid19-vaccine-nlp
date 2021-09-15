
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from batch_utils import get_collator_type


class BaseDataModule(pl.LightningDataModule):
	def __init__(
			self,
			tokenizer_name: str,
			train_path: str = None,
			val_path: str = None,
			test_path: str = None,
			predict_path: str = None,
			collator_type: str = 'multi_sequence',
			batch_size: int = 32,
			max_seq_len: int = 512,
			num_workers: int = 8,
			use_tpus: bool = False,
	):
		super().__init__()

		self.tokenizer_name = tokenizer_name
		self.collator_type = collator_type
		self.batch_size = batch_size
		self.max_seq_len = max_seq_len
		self.num_workers = num_workers
		self.use_tpus = use_tpus
		self.tokenizer = AutoTokenizer.from_pretrained(
			self.tokenizer_name
		)

		self.collator = get_collator_type(collator_type)

		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path
		self.predict_path = predict_path

		self.train_dataset = None
		self.val_dataset = None
		self.test_dataset = None
		self.predict_dataset = None

	def train_dataloader(self):
		train_dataloader = DataLoader(
			self.train_dataset,
			num_workers=self.num_workers,
			batch_size=self.batch_size,
			shuffle=True,
			drop_last=True,
			collate_fn=self.collator(
				max_seq_len=self.max_seq_len,
				use_tpus=self.use_tpus,
			),
			worker_init_fn=self.train_dataset.worker_init_fn
		)
		return train_dataloader

	def val_dataloader(self):
		val_dataloader = DataLoader(
			self.val_dataset,
			num_workers=self.num_workers,
			batch_size=self.batch_size,
			shuffle=False,
			collate_fn=self.collator(
				max_seq_len=self.max_seq_len,
				use_tpus=self.use_tpus,
			),
			worker_init_fn=self.val_dataset.worker_init_fn
		)
		return val_dataloader

	def test_dataloader(self):
		test_dataloader = DataLoader(
			self.test_dataset,
			num_workers=self.num_workers,
			batch_size=self.batch_size,
			shuffle=False,
			collate_fn=self.collator(
				max_seq_len=self.max_seq_len,
				use_tpus=self.use_tpus,
			),
			worker_init_fn=self.test_dataset.worker_init_fn
		)
		return test_dataloader

	def predict_dataloader(self):
		predict_dataloader = DataLoader(
			self.predict_dataset,
			num_workers=self.num_workers,
			batch_size=self.batch_size,
			shuffle=False,
			collate_fn=self.collator(
				max_seq_len=self.max_seq_len,
				use_tpus=self.use_tpus,
			),
			worker_init_fn=self.predict_dataset.worker_init_fn
		)
		return predict_dataloader
