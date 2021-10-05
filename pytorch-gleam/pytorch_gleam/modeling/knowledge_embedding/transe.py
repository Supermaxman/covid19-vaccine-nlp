
from torch import nn
import torch

from pytorch_gleam.modeling.knowledge_embedding.base_emb import KnowledgeEmbedding


class TransEEmbedding(KnowledgeEmbedding):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.e_emb_layer = nn.Linear(
			self.hidden_size,
			self.emb_size
		)
		self.r_emb_layer = nn.Linear(
			self.hidden_size,
			self.emb_size
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
		h_r_t_diff = head + rel - tail
		return self.diff_energy(h_r_t_diff)

	def loss(self, pos_energy, neg_energy):
		return self.margin_loss(pos_energy, neg_energy)
