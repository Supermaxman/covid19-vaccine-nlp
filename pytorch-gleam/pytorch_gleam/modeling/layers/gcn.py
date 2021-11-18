
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
	def __init__(self, in_features, out_features, bias=True, init='xavier'):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		if init == 'uniform':
			print("| Uniform Initialization")
			self.reset_parameters_uniform()
		elif init == 'xavier':
			print("| Xavier Initialization")
			self.reset_parameters_xavier()
		elif init == 'kaiming':
			print("| Kaiming Initialization")
			self.reset_parameters_kaiming()
		else:
			raise NotImplementedError

	def reset_parameters_uniform(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def reset_parameters_xavier(self):
		nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform
		if self.bias is not None:
			nn.init.constant_(self.bias.data, 0.0)

	def reset_parameters_kaiming(self):
		nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
		if self.bias is not None:
			nn.init.constant_(self.bias.data, 0.0)

	def forward(self, inputs, adj):
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, hidden_size]
		# support = torch.mm(inputs, self.weight)
		support = torch.matmul(inputs, self.weight)
		# [bsize, seq_len, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, seq_len, hidden_size]
		# output = torch.mm(adj, support)
		output = torch.matmul(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
					 + str(self.in_features) + ' -> ' \
					 + str(self.out_features) + ')'


class GraphAttention(nn.Module):
	def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True, return_attention=False):
		super(GraphAttention, self).__init__()
		self.dropout = dropout
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat
		self.return_attention = return_attention

		self.W = nn.Parameter(
			nn.init.xavier_normal_(
				torch.Tensor(in_features, out_features), gain=np.sqrt(2.0)),
			requires_grad=True)
		self.a1 = nn.Parameter(
			nn.init.xavier_normal_(
				torch.Tensor(out_features, 1), gain=np.sqrt(2.0)),
			requires_grad=True)
		self.a2 = nn.Parameter(
			nn.init.xavier_normal_(
				torch.Tensor(out_features, 1), gain=np.sqrt(2.0)),
			requires_grad=True)

		self.leakyrelu = nn.LeakyReLU(self.alpha)

	def forward(self, inputs, adj):
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, hidden_size]
		h = torch.matmul(inputs, self.W)
		# [bsize, seq_len, hidden_size] x [hidden_size, 1] -> [bsize, seq_len, 1]
		f_1 = torch.matmul(h, self.a1)
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, 1]
		f_2 = torch.matmul(h, self.a2)
		# [bsize, seq_len, 1] + [bsize, 1, seq_len] -> [bsize, seq_len, seq_len]
		e = self.leakyrelu(f_1 + f_2.transpose(-2, -1))
		# [bsize, seq_len, seq_len]
		zero_vec = -9e15 * torch.ones_like(e)
		# [bsize, seq_len, seq_len]
		attention = torch.where(adj > 0, e, zero_vec)
		attention = F.softmax(attention, dim=-1)
		attention = F.dropout(attention, self.dropout, training=self.training)
		# [bsize, seq_len, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, seq_len, hidden_size]
		h_prime = torch.matmul(attention, h)

		if self.concat:
			return_vals = F.elu(h_prime)
		else:
			return_vals = h_prime
		if self.return_attention:
			return_vals = (return_vals, attention)

		return return_vals

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class TransformerGraphAttention(nn.Module):
	def __init__(self, in_features, out_features, dropout_prob, activation=True):
		super().__init__()
		self.dropout = nn.Dropout(dropout_prob)
		self.normalizer = nn.Softmax(dim=-1)
		self.in_features = in_features
		self.out_features = out_features
		self.activation = activation

		self.query = nn.Linear(in_features, out_features)
		self.key = nn.Linear(in_features, out_features)
		self.value = nn.Linear(in_features, out_features)

	def forward(self, inputs, adj):
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, hidden_size]
		q = self.query(inputs)
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, hidden_size]
		k = self.key(inputs)
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, hidden_size]
		v = self.value(inputs)
		# [bsize, seq_len, seq_len]
		adj = adj.float()
		adj = (1.0 - adj) * -10000.0

		# [bsize, seq_len, hidden_size] x [bsize, hidden_size, seq_len] -> [bsize, seq_len, seq_len]
		a = torch.matmul(q, k.transpose(-1, -2))
		a = a / math.sqrt(q.shape[-1])
		# [bsize, seq_len, seq_len] + [bsize, seq_len, seq_len]
		a = a + adj
		a_probs = self.normalizer(a)
		a_probs = self.dropout(a_probs)
		# [bsize, seq_len, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, seq_len, hidden_size]
		h_prime = torch.matmul(a_probs, v)

		if self.activation:
			return F.elu(h_prime)
		else:
			return h_prime

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# TODO multi-head graph attention
class EdgeTransformerGraphAttention(nn.Module):
	def __init__(self, in_features, out_features, dropout_prob, activation=True):
		super().__init__()
		self.dropout = nn.Dropout(dropout_prob)
		self.normalizer = nn.Softmax(dim=-1)
		self.in_features = in_features
		self.out_features = out_features
		self.activation = activation

		self.query = nn.Linear(in_features, out_features)
		self.key = nn.Linear(in_features, out_features)
		self.value = nn.Linear(in_features, out_features)

	def forward(self, inputs, adj):
		# TODO
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, hidden_size]
		q = self.query(inputs)
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, hidden_size]
		k = self.key(inputs)
		# [bsize, seq_len, hidden_size] x [hidden_size, hidden_size] -> [bsize, seq_len, hidden_size]
		v = self.value(inputs)

		adj = adj.float()
		adj = adj.unsqueeze(dim=1)
		adj = (1.0 - adj) * -10000.0

		# [bsize, seq_len, hidden_size] x [bsize, hidden_size, seq_len] -> [bsize, seq_len, seq_len]
		a = torch.matmul(q, k.transpose(-1, -2))
		a = a / math.sqrt(q.shape[-1])
		a = a + adj
		a_probs = self.normalizer(a)
		a_probs = self.dropout(a_probs)
		# [bsize, seq_len, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, seq_len, hidden_size]
		h_prime = torch.matmul(a_probs, v)

		if self.activation:
			return F.elu(h_prime)
		else:
			return h_prime

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
