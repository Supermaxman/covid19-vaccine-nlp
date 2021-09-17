
import math
from torch import nn
import torch


class RotatEEmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size, gamma, loss_norm=2):
		super().__init__()
		self.gamma = gamma
		self.emb_size = emb_size
		self.loss_norm = loss_norm
		self.td_emb_size = self.emb_size // 2
		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)
		self.e_proj_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)
		self.r_emb_layer = nn.Linear(
			hidden_size,
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

		# l2 norm squared = sum of squares
		if self.loss_norm == 1:
			h_r_t_energy = torch.norm(h_r_t_diff, p=1, dim=-1, keepdim=False)
		elif self.loss_norm == 2:
			h_r_t_energy = (h_r_t_diff * h_r_t_diff).sum(dim=-1)
		else:
			raise ValueError(f'Unknown loss norm: {self.loss_norm}')
		return h_r_t_energy

	def loss(self, pos_energy, neg_energy):
		pos_loss = -torch.log(torch.sigmoid(self.gamma - pos_energy) + 1e-6)
		neg_loss = -torch.log(torch.sigmoid(neg_energy - self.gamma) + 1e-6)
		loss = pos_loss + neg_loss
		accuracy = (pos_energy.lt(neg_energy)).float().mean()
		return loss, accuracy
