from abc import abstractmethod

from torch import nn
import torch


class KnowledgeEmbedding(nn.Module):
	def __init__(self, hidden_size: int, emb_size: int, gamma: float, loss_norm: int = 2):
		super().__init__()
		self.gamma = gamma
		self.hidden_size = hidden_size
		self.emb_size = emb_size
		self.loss_norm = loss_norm

	@abstractmethod
	def forward(self, source_embeddings, emb_type):
		pass

	def diff_energy(self, diff):
		if self.loss_norm == 1:
			h_r_t_energy = torch.norm(diff, p=1, dim=-1, keepdim=False)
		elif self.loss_norm == 2:
			h_r_t_energy = (diff * diff).sum(dim=-1)
		else:
			raise ValueError(f'Unknown loss norm: {self.loss_norm}')
		return h_r_t_energy

	@abstractmethod
	def energy(self, head, rel, tail):
		pass

	@abstractmethod
	def loss(self, pos_energy, neg_energy):
		pass

	def margin_loss(self, pos_energy, neg_energy):
		# [bsize, pos_samples]
		# pos_energy
		# [bsize, neg_samples]
		# neg_energy
		# [bsize, 1, pos_samples]
		pos_energy = pos_energy.unsqueeze(dim=-2)
		# [bsize, neg_samples, 1]
		neg_energy = neg_energy.unsqueeze(dim=-1)
		# [bsize, neg_samples, pos_samples]
		margin = pos_energy - neg_energy
		loss = torch.clamp(self.gamma + margin, min=0.0)
		accuracy = (pos_energy.lt(neg_energy)).float()
		return loss, accuracy
