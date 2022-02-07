
from torch import nn
import torch

from pytorch_gleam.modeling.knowledge_embedding.base_emb import KnowledgeEmbedding


class TransDEmbedding(KnowledgeEmbedding):
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
		self.r_proj_layer = nn.Linear(
			self.hidden_size,
			self.td_emb_size
		)

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			ex_projs = self.e_proj_layer(source_embeddings)
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			ex_embs = self.r_emb_layer(source_embeddings)
			ex_projs = self.r_proj_layer(source_embeddings)
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')
		# https://www.aclweb.org/anthology/P15-1067.pdf
		# normalize all lookups to max l2 norm of 1

		# [bsize * num_seq, emb_size]
		# ex_embs = ex_embs / torch.clamp(ex_emb_norms, min=1.0)
		ex_emb_norms = torch.norm(ex_embs, p=2, dim=-1, keepdim=True)
		ex_embs = ex_embs / ex_emb_norms

		ex_embs = torch.cat([ex_embs, ex_projs], dim=-1)
		return ex_embs

	@staticmethod
	def project(c, c_proj, r_proj):
		c_p = c + torch.sum(c * c_proj, dim=-1, keepdim=True) * r_proj
		c_p_norm = torch.norm(c_p, p=2, dim=-1, keepdim=True)
		# c_p = c_p / torch.clamp(c_p_norm, min=1.0)
		c_p = c_p / c_p_norm
		return c_p

	def energy(self, head, rel, tail):
		h, h_proj = head[..., :self.td_emb_size], head[..., self.td_emb_size:]
		r, r_proj = rel[..., :self.td_emb_size], rel[..., self.td_emb_size:]
		t, t_proj = tail[..., :self.td_emb_size], tail[..., self.td_emb_size:]
		h_p = self.project(h, h_proj, r_proj)
		t_p = self.project(t, t_proj, r_proj)
		h_r_t_diff = h_p + r - t_p
		return self.diff_energy(h_r_t_diff)

	def loss(self, pos_energy, neg_energy):
		return self.margin_loss(pos_energy, neg_energy)

