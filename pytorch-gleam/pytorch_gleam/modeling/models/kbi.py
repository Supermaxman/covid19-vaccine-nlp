
import torch

from pytorch_gleam.modeling.base_models import BaseLanguageModel
from pytorch_gleam.modeling.knowledge_embedding import *


# noinspection PyAbstractClass
class KbiLanguageModel(BaseLanguageModel):
	def __init__(
			self,
			ke_model: str = 'transms',
			ke_emb_size: int = 8,
			ke_hidden_size: int = 32,
			ke_gamma: float = 1.0,
			ke_loss_norm: int = 1,
			num_relations: int = 2,
			metric: str = 'f1',
			metric_mode: str = 'macro',
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.num_relations = num_relations
		self.ke_model = ke_model
		self.ke_emb_size = ke_emb_size
		self.ke_hidden_size = ke_hidden_size
		self.ke_gamma = ke_gamma
		self.ke_loss_norm = ke_loss_norm
		self.ke_rel_layers = torch.nn.ModuleList(
			[
				torch.nn.Linear(
					in_features=self.hidden_size,
					out_features=self.ke_hidden_size
				) for _ in range(self.num_relations)
			]
		)
		self.ke_entity_layer = torch.nn.Linear(
			in_features=self.hidden_size,
			out_features=self.ke_hidden_size
		)
		# TODO select ke from self.ke_model
		self.ke = TransMSEmbedding(
			hidden_size=self.ke_hidden_size,
			emb_size=self.ke_emb_size,
			gamma=self.ke_gamma,
			loss_norm=self.ke_loss_norm
		)
		# TODO build multi-class multi-label threshold module
		# self.threshold = MultiClassThresholdModule()
		# TODO select based on metric
		# self.metric = F1PRMultiClassMetric(
		# 	num_classes=self.num_classes,
		# 	mode=metric_mode
		# )
	# TODO
	# def predict_step(self, batch, batch_idx, dataloader_idx=None):
	#

	def eval_epoch_end(self, outputs, stage):
		loss = torch.cat([x['loss'] for x in outputs], dim=0).mean()
		accuracy = torch.cat([x['accuracy'] for x in outputs], dim=0).mean()

		self.log(f'{stage}_loss', loss)
		self.log(f'{stage}_accuracy', accuracy)

	def eval_step(self, batch, batch_idx, dataloader_idx=None):
		loss, accuracy = self.triplet_step(batch)
		result = {
			'loss': loss,
			'accuracy': accuracy,
		}
		return result

	def forward(self, batch):
		num_examples = batch['num_examples']
		num_sequences_per_example = batch['num_sequences_per_example']
		num_entities = num_sequences_per_example - 1
		pad_seq_len = batch['pad_seq_len']

		# [bsize, num_seq, seq_len] -> [bsize * num_seq, seq_len]
		input_ids = batch['input_ids'].view(num_examples * num_sequences_per_example, pad_seq_len)
		attention_mask = batch['attention_mask'].view(num_examples * num_sequences_per_example, pad_seq_len)
		token_type_ids = batch['token_type_ids'].view(num_examples * num_sequences_per_example, pad_seq_len)

		# [bsize * num_seq, seq_len, hidden_size]
		contextualized_embeddings = self.lm_step(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		# [bsize * num_seq, hidden_size]
		lm_output = contextualized_embeddings[:, 0]
		# TODO consider dropout
		# lm_output = self.f_dropout(lm_output)
		lm_output = lm_output.view(num_examples, num_sequences_per_example, self.hidden_size)
		# [bsize, hidden_size]
		r_lm_output = lm_output[:, 0]
		# [bsize * num_entities, hidden_size]
		e_lm_output = lm_output[:, 1:].reshape(num_examples * num_entities, self.hidden_size)
		e_proj = self.ke_entity_layer(e_lm_output)
		e_embs = self.ke(e_proj, 'entity')
		# [bsize, num_entities, emb_size]
		e_embs = e_embs.view(num_examples, num_entities, e_embs.shape[-1])
		# num_samples = batch['pos_samples'] + batch['neg_samples']
		# [bsize, num_samples, num_relations]
		# relation_mask = batch['relation_mask']
		r_projections = []
		for r_layer in self.ke_rel_layers:
			r_proj = r_layer(r_lm_output)
			r_projections.append(r_proj)
		# [bsize, num_relations, ke_hidden_size]
		r_projections = torch.stack(r_projections, dim=1)
		# [bsize * num_relations, ke_hidden_size]
		r_projections = r_projections.view(num_examples * self.num_relations, self.ke_hidden_size)
		# [bsize * num_relations, ke_emb_size]
		r_embs = self.ke(r_projections, 'rel')
		# [bsize, num_relations, ke_emb_size]
		r_embs = r_embs.view(num_examples, self.num_relations, r_embs.shape[-1])
		# [bsize, num_entities, emb_size]
		return e_embs, r_embs

	@staticmethod
	def split_embeddings(embs, batch):
		t_ex_embs = embs[:, 0]
		pos_samples = batch['pos_samples']
		if pos_samples > 0:
			pos_embs = embs[:, 1:1+pos_samples]
		else:
			pos_embs = None
		neg_samples = batch['neg_samples']
		if neg_samples > 0:
			neg_embs = embs[:, 1+pos_samples:1+pos_samples+neg_samples]
		else:
			neg_embs = None
		return t_ex_embs, pos_embs, neg_embs

	def triplet_energy(self, e_embs, m_embs, batch):
		# all
		# t_ex_embs: [bsize, emb_size],
		# pos_embs: [bsize, pos_samples, emb_size],
		# neg_embs: [bsize, pos_samples, emb_size],
		t_ex_embs, pos_embs, neg_embs = self.split_embeddings(e_embs, batch)
		# [bsize, 1, emb_size]
		t_ex_embs = t_ex_embs.unsqueeze(dim=-2)
		# [bsize, 1, num_relations, emb_size]
		m_embs = m_embs.unsqueeze(dim=-3)
		# [bsize, pos_samples + neg_samples, num_relations, 1]
		rel_mask = batch['relation_mask'].unsqueeze(dim=-1)
		# [bsize, pos_samples + neg_samples, num_relations, emb_size]
		m_embs = m_embs * rel_mask
		pos_samples = batch['pos_samples']
		neg_samples = batch['neg_samples']
		# [bsize, pos_samples + neg_samples, emb_size]
		# using the rel_mask we are able to sum over m_embs that are zero'd
		m_embs = m_embs.sum(dim=-2)
		# [bsize, pos_samples, emb_size]
		pos_m_embs = m_embs[..., :pos_samples, :]
		# [bsize, neg_samples, emb_size]
		neg_m_embs = m_embs[..., pos_samples:pos_samples+neg_samples, :]

		# [bsize]
		pos_forward_energy = self.ke.energy(t_ex_embs, pos_m_embs, pos_embs)
		# [bsize]
		pos_backward_energy = self.ke.energy(pos_embs, pos_m_embs, t_ex_embs)
		# [bsize, 2]
		pos_energy = torch.stack([pos_forward_energy, pos_backward_energy], dim=-1)
		# [bsize]
		neg_forward_energy = self.ke.energy(t_ex_embs, neg_m_embs, neg_embs)
		# [bsize]
		neg_backward_energy = self.ke.energy(neg_embs, neg_m_embs, t_ex_embs)
		# [bsize, 2]
		neg_energy = torch.stack([neg_forward_energy, neg_backward_energy], dim=-1)
		direction_mask = batch['direction_mask']
		# first randomly pick between subject and object losses
		# [bsize]
		pos_energy = (pos_energy * direction_mask).sum(dim=-1)
		# [bsize]
		neg_energy = (neg_energy * direction_mask).sum(dim=-1)

		return pos_energy, neg_energy

	def loss(self, pos_energy, neg_energy):
		loss, accuracy = self.ke.loss(pos_energy, neg_energy)
		loss = loss.mean()
		return loss, accuracy

	def triplet_step(self, batch):
		e_embs, r_embs = self(batch)
		pos_energy, neg_energy = self.triplet_energy(e_embs, r_embs, batch)
		loss, accuracy = self.loss(pos_energy, neg_energy)
		return loss, accuracy

	def training_step(self, batch, batch_idx):
		loss, accuracy = self.triplet_step(batch)
		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
		result = {
			'loss': loss
		}
		return result
