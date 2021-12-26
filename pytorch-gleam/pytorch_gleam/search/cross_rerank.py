
import os
import json
import argparse
import logging
import pytorch_lightning as pl
from torch.utils.data import IterableDataset
from transformers import BertTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.distributed as dist


def get_tweet_text(tweet):
	tweet_text = tweet['text']
	if tweet_text.startswith('RT '):
		ref_tweets = tweet['referenced_tweets']
		if len(ref_tweets) > 0:
			rt_data = ref_tweets[0]['data']
			if 'text' in rt_data:
				tweet_text = rt_data['text']
	if 'entities' in tweet:
		for e_type, e_list in tweet['entities'].items():
			if e_type == 'urls':
				for e_url in e_list:
					r_url = e_url['url']
					s_url = e_url['expanded_url']
					tweet_text = tweet_text.replace(r_url, s_url)
	return tweet_text


def read_jsonl(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					ex = json.loads(line)
					yield ex
				except Exception as e:
					print(e)


def get_tweets(file_path):
	for ex in read_jsonl(file_path):
		if 'tweet' in ex:
			ex = ex['tweet']
		yield ex


def worker_init_fn(_):
	# ISSUE: this only works for WORKERS within the same process, not
	# TODO multiprocessing
	process_id = dist.get_rank()
	num_processes = dist.get_world_size()

	worker_info = torch.utils.data.get_worker_info()
	worker_id = worker_info.id
	num_workers = worker_info.num_workers
	print(f'INFO: WORKER_INIT WORKER_INFO: {worker_id}/{num_workers}')
	print(f'INFO: WORKER_INIT: RANK_INFO: {process_id}/{num_processes}')
	dataset = worker_info.dataset
	# dataset.frequency = worker_id
	# dataset.num_workers = num_workers
	dataset.frequency = (process_id * num_workers) + worker_id
	dataset.num_workers = num_processes * num_workers
	print(f'INFO: WORKER_INIT: F_INFO: {dataset.frequency}/{dataset.num_workers}')


class RerankDataset(IterableDataset):
	def __init__(self, data_path, questions_path, worker_estimate=6):

		with open(questions_path, 'r') as f:
			self.questions = json.load(f)

		self.data_path = data_path
		self.frequency = 0
		self.num_workers = 1
		self.tweet_examples = defaultdict(list)
		self.num_examples = 0
		self.worker_estimate = worker_estimate
		for tweet in get_tweets(self.data_path):
			tweet_id = tweet['id']
			ignore_q_ids = set()
			for q_id, q in tweet['candidates'].items():
				ignore_q_ids.add(q_id)
			for q_id in self.questions:
				self.tweet_examples[tweet_id].append(q_id)
				self.num_examples += 1
		print(f'Num examples: {self.num_examples}')

	def __len__(self):
		return self.num_examples // self.worker_estimate

	def __iter__(self):
		ex_idx = 0
		for tweet in get_tweets(self.data_path):
			tweet_id = tweet['id']
			if tweet_id not in self.tweet_examples:
				continue
			tweet_text = get_tweet_text(tweet)
			# only do relevance re-ranking on bm25 results
			q_exs = self.tweet_examples[tweet_id]
			# OR do relevance re-ranking on all results (EXPENSIVE)
			# q_exs = self.questions
			for q_id in q_exs:
				ex = {
					'id': tweet_id,
					'question_id': f'{q_id}',
					'query': self.questions[q_id]['text'],
					'text': tweet_text
				}
				if ex_idx % self.num_workers == self.frequency:
					yield ex
				ex_idx += 1


class RerankBatchCollator(object):
	def __init__(self, tokenizer, max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, examples):
		ids = []
		question_ids = []
		sequences = []
		for ex in examples:
			ids.append(ex['id'])
			question_ids.append(ex['question_id'])
			sequences.append((ex['query'], ex['text']))

		tokenizer_batch = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs=sequences,
			add_special_tokens=True,
			padding='max_length' if self.force_max_seq_len else 'longest',
			return_tensors='pt',
			truncation='only_second',
			max_length=self.max_seq_len
		)
		batch = {
			'id': ids,
			'question_id': question_ids,
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask'],
			'token_type_ids': tokenizer_batch['token_type_ids'],
		}

		return batch


class RerankBert(pl.LightningModule):
	def __init__(
			self, pre_model_name, predict_mode=False, predict_path=None):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.predict_mode = predict_mode
		self.predict_path = predict_path
		self.bert = AutoModelForSequenceClassification.from_pretrained(
			pre_model_name
		)
		self.config = self.bert.config
		self.save_hyperparameters()

	def forward(self, input_ids, attention_mask, token_type_ids):
		# [batch_size, 2]
		logits = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0]
		return logits

	def training_step(self, batch, batch_nb):
		pass

	def test_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'test')

	def validation_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'val')

	def _forward_step(self, batch, batch_nb):
		logits = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)

		return logits

	def _eval_step(self, batch, batch_nb, name):
		logits = self._forward_step(batch, batch_nb)
		logits = logits.detach().cpu()
		device_id = get_device_id()
		self.write_prediction_dict(
			{
				'id': batch['id'],
				'question_id': batch['question_id'],
				'pos_score': logits[:, 1].tolist(),
				'neg_score': logits[:, 0].tolist(),
			},
			filename=os.path.join(self.predict_path, f'predictions-{device_id}.pt')
		)
		result = {
			f'{name}_id': batch['id'],
			f'{name}_question_id': batch['question_id'],
			f'{name}_logits': logits,
		}

		return result

	def _eval_epoch_end(self, outputs, name):
		pass

	def validation_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'test')


def get_device_id():
	try:
		device_id = dist.get_rank()
	except Exception:
		if 'XRT_SHARD_ORDINAL' in os.environ:
			device_id = int(os.environ['XRT_SHARD_ORDINAL'])
		else:
			device_id = 0
	return device_id


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--data_path', required=True)
	parser.add_argument('-qp', '--questions_path', default=None)
	parser.add_argument('-op', '--output_path', required=True)
	parser.add_argument('-pm', '--pre_model_name', default='nboost/pt-biobert-base-msmarco')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-bs', '--batch_size', default=4, type=int)
	parser.add_argument('-ml', '--max_seq_len', default=96, type=int)
	parser.add_argument('-se', '--seed', default=0, type=int)
	parser.add_argument('-cd', '--torch_cache_dir', default=None)
	parser.add_argument('-tpu', '--use_tpus', default=False, action='store_true')
	parser.add_argument('-gpu', '--gpus', default='0')
	parser.add_argument('-ts', '--train_sampling', default='none')
	parser.add_argument('-ls', '--losses', default='compare_loss')

	args = parser.parse_args()

	pl.seed_everything(args.seed)

	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	gpus = [int(x) for x in args.gpus.split(',')]
	is_distributed = len(gpus) > 1
	precision = 16 if args.use_tpus else 32
	# precision = 32
	tpu_cores = 8
	num_workers = 1
	deterministic = True

	tokenizer = BertTokenizer.from_pretrained(args.pre_model_name)

	logging.info('Loading datasets...')

	val_dataset = RerankDataset(
		data_path=args.data_path,
		questions_path=args.questions_path,
		worker_estimate=len(gpus)
	)
	val_data_loader = DataLoader(
		val_dataset,
		num_workers=num_workers,
		shuffle=False,
		batch_size=args.batch_size,
		collate_fn=RerankBatchCollator(
			tokenizer,
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		),
		worker_init_fn=worker_init_fn
	)

	logging.info('Loading model...')

	model = RerankBert(
		pre_model_name=args.pre_model_name,
		predict_mode=True,
		predict_path=args.output_path
	)

	if args.use_tpus:
		logging.warning('Gradient clipping slows down TPU training drastically, disabled for now.')
		trainer = pl.Trainer(
			tpu_cores=tpu_cores,
			max_epochs=0,
			precision=precision,
			deterministic=deterministic,
			checkpoint_callback=False,
		)
	else:
		if len(gpus) > 1:
			backend = 'ddp' if is_distributed else 'dp'
		else:
			backend = None
		trainer = pl.Trainer(
			gpus=gpus,
			max_epochs=0,
			precision=precision,
			distributed_backend=backend,
			deterministic=deterministic,
			checkpoint_callback=False,
		)

	logging.info('Predicting...')
	try:
		trainer.test(model, val_data_loader)
	except Exception as e:
		logging.exception('Exception during predicting', exc_info=e)


if __name__ == '__main__':
	main()
