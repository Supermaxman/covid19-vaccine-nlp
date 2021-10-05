
import math
from torch import nn
import torch

from pytorch_gleam.modeling.knowledge_embedding.base_emb import KnowledgeEmbedding


class RotatEEmbedding(KnowledgeEmbedding):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.td_emb_size = self.emb_size // 2
		self.e_emb_layer = nn.Linear(
			self.hidden_size,
			self.td_emb_size
		)
		self.e_proj_layer = nn.Linear(
			self.hidden_size,
			self.td_emb_size
		)
		self.r_emb_layer = nn.Linear(
			self.hidden_size,
			self.td_emb_size
		)

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			ex_ims = self.e_proj_layer(source_embeddings)
			ex_embs = torch.cat([ex_embs, ex_ims], dim=-1)
			return ex_embs
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			r_embs = self.r_emb_layer(source_embeddings)
			return r_embs
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')

	def energy(self, head, rel, tail):
		h_re, h_im = head[..., :self.td_emb_size], head[..., self.td_emb_size:]
		t_re, t_im = tail[..., :self.td_emb_size], tail[..., self.td_emb_size:]
		r_phase = torch.tanh(rel) * math.pi
		r_re = torch.cos(r_phase)
		r_im = torch.sin(r_phase)

		re_score = (h_re * r_re - h_im * r_im) - t_re
		im_score = (h_re * r_im + h_im * r_re) - t_im
		h_r_t_diff = torch.cat([re_score, im_score], dim=-1)

		return self.diff_energy(h_r_t_diff)

	def loss(self, pos_energy, neg_energy):
		pos_loss = -torch.log(torch.sigmoid(self.gamma - pos_energy) + 1e-6)
		neg_loss = -torch.log(torch.sigmoid(neg_energy - self.gamma) + 1e-6)
		loss = pos_loss + neg_loss
		accuracy = (pos_energy.lt(neg_energy)).float()
		return loss, accuracy
