
from torch import nn
import torch


class TransMSEmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size, gamma, loss_norm=2):
		super().__init__()
		self.gamma = gamma
		self.emb_size = emb_size
		self.loss_norm = loss_norm
		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)
		self.r_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size + 1
		)

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			ex_emb_norms = torch.norm(ex_embs, p=2, dim=-1, keepdim=True)
			# [bsize * num_seq, emb_size]
			ex_embs = ex_embs / ex_emb_norms
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			ex_embs = self.r_emb_layer(source_embeddings)
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')

		return ex_embs

	def energy(self, head, rel, tail):
		rel = rel[..., :self.emb_size]
		alpha = rel[..., -1]
		alpha = alpha.unsqueeze(dim=-1)
		h_p = -torch.tanh(tail * rel) * head
		r_p = rel + alpha * (head * tail)
		t_p = torch.tanh(head * rel) * tail

		h_r_t_diff = h_p + r_p - t_p
		if self.loss_norm == 1:
			h_r_t_energy = torch.norm(h_r_t_diff, p=1, dim=-1, keepdim=False)
		elif self.loss_norm == 2:
			h_r_t_energy = (h_r_t_diff * h_r_t_diff).sum(dim=-1)
		else:
			raise ValueError(f'Unknown loss norm: {self.loss_norm}')
		return h_r_t_energy

	def loss(self, pos_energy, neg_energy):
		margin = pos_energy - neg_energy
		loss = torch.clamp(self.gamma + margin, min=0.0)
		accuracy = (pos_energy.lt(neg_energy)).float()
		return loss, accuracy
