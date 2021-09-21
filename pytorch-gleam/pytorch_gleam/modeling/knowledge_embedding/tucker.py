
from torch import nn
import torch
import numpy as np


class TuckEREmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size, gamma, loss_norm=2):
		super().__init__()
		self.gamma = gamma
		self.emb_size = emb_size
		self.loss_norm = loss_norm

		self.weight = nn.parameter.Parameter(
			torch.tensor(np.random.uniform(-1, 1, (self.emb_size, self.emb_size, self.emb_size)))
		)

		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)
		self.r_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)
		# self.score_func = nn.LogSigmoid()

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			return ex_embs
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			r_embs = self.r_emb_layer(source_embeddings)
			return r_embs
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')

	def energy(self, head, rel, tail):
		num_batch_dims = len(head.shape) - 1
		w = self.weight
		# [1, 1, ..., e_size, e_size, e_size]
		for _ in range(num_batch_dims):
			w = w.unsqueeze(dim=0)
		head = head.unsqueeze(dim=-1)
		head = head.unsqueeze(dim=-1)
		# [..., emb_size, emb_size, emb_size] -> [..., emb_size, emb_size]
		w = (w * head).sum(dim=-1)
		# [..., emb_size, 1]
		rel = rel.unsqueeze(dim=-1)
		# [..., emb_size]
		w = (w * rel).sum(dim=-1)
		# [...]
		w = (w * tail).sum(dim=-1)
		# this is treated as a score by TuckER, so - makes energy
		h_r_t_energy = -w
		return h_r_t_energy

	def loss(self, pos_energy, neg_energy):

		pos_loss = -torch.log(torch.sigmoid(-pos_energy) + 1e-6)
		neg_loss = -torch.log(1.0 - torch.sigmoid(-neg_energy) + 1e-6)
		loss = pos_loss + neg_loss
		accuracy = (pos_energy.lt(neg_energy)).float()
		return loss, accuracy
