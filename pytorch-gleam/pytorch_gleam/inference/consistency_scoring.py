
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import networkx as nx
import numpy as np
import torch
from torch import nn

from pytorch_gleam.data.datasets.kbi_misinfo_stance import flip_tm_stance


class ConsistencyScoring(nn.Module, ABC):
	def __init__(self):
		super().__init__()

	@abstractmethod
	def forward(
			self,
			adj_list: List[Tuple[str, str, Tuple[float, float]]],
			node_labels: Dict[str, int]
	) -> Tuple[np.array, Dict[str, int]]:
		pass


class MultiHopConsistencyScoring(ConsistencyScoring):
	def __init__(self, num_steps: int = 1, num_classes: int = 3):
		super().__init__()
		self.num_steps = num_steps
		self.num_classes = num_classes

	def initialize(self, adj_list, seed_node_labels):
		g = nx.Graph()
		# list of (ex_t_id, ex_p_id, ex_tmp_energy)
		# 0 - entail
		# 1 - contradict
		nodes = set()
		unlabeled_nodes = []
		labeled_nodes = []
		node_idx = {}
		for t_id, p_id, tp_r_dists in adj_list:
			if t_id not in nodes:
				node_idx[t_id] = len(node_idx)
				if t_id in seed_node_labels:
					labeled_nodes.append(t_id)
				else:
					unlabeled_nodes.append(t_id)
			if p_id not in nodes:
				node_idx[p_id] = len(node_idx)
				if p_id in seed_node_labels:
					labeled_nodes.append(p_id)
				else:
					unlabeled_nodes.append(p_id)

			nodes.add(t_id)
			nodes.add(p_id)
			entail_weight, contradict_weight = tp_r_dists
			entail_weight = -entail_weight.item()
			contradict_weight = -contradict_weight.item()
			g.add_edge(t_id, p_id, entail_weight=entail_weight, contradict_weight=contradict_weight)
		return g, unlabeled_nodes, labeled_nodes, node_idx

	def propagate_seeds(self, labeled_nodes, unlabeled_nodes, graph, nls, nlc, seed_node_labels, node_idx):
		for node in unlabeled_nodes:
			for other_node in labeled_nodes:
				other_label = seed_node_labels[other_node]
				if isinstance(other_label, torch.Tensor):
					other_label = other_label.item()
				# entailment, contradiction, or neither does not mean anything if we know
				# the label of the other node
				if other_label == 0:
					continue
				edge = graph.get_edge_data(node, other_node)
				entail_score = edge['entail_weight']
				entail_label = other_label
				contradict_score = edge['contradict_weight']
				contradict_label = flip_tm_stance(other_label)
				n_idx = node_idx[node]
				nls[n_idx, entail_label, 0] += entail_score
				nlc[n_idx, entail_label, 0] += 1
				nls[n_idx, contradict_label, 0] += contradict_score
				nlc[n_idx, contradict_label, 0] += 1
		nls[:, :, 0] = nls[:, :, 0] / nlc[:, :, 0]

	def propagate(self, nodes, graph, nls, nlc, step, node_idx):
		for node in nodes:
			for other_node in nodes:
				if node == other_node:
					continue
				edge = graph.get_edge_data(node, other_node)
				entail_score = edge['entail_weight']
				contradict_score = edge['contradict_weight']
				n_idx = node_idx[node]
				o_idx = node_idx[other_node]
				for other_pred in [1, 2]:
					entail_label = other_pred
					contradict_label = flip_tm_stance(other_pred)
					node_score = nls[o_idx, other_pred, step - 1]
					# node_count = nlc[o_idx, other_pred, step-1]
					nls[n_idx, entail_label, step] += (node_score + entail_score)
					nlc[n_idx, entail_label, step] += 1
					nls[n_idx, contradict_label, step] += (node_score + contradict_score)
					nlc[n_idx, contradict_label, step] += 1
		nls[:, :, step] = nls[:, :, step] / nlc[:, :, step]

	def forward(
			self,
			adj_list: List[Tuple[str, str, Tuple[float, float]]],
			node_labels: Dict[str, int]
	) -> Tuple[np.array, Dict[str, int]]:
		assert len(node_labels) > 0

		graph, unlabeled_nodes, labeled_nodes, node_idx = self.initialize(adj_list, node_labels)
		num_steps = self.num_steps
		# no need to do any propagation steps if there are no paths in unlabeled graph
		if len(unlabeled_nodes) == 1:
			num_steps = 0

		# [num_nodes, num_labels, num_steps]
		nls = np.zeros([len(node_idx), self.num_classes, num_steps + 1], dtype=np.float32)
		nlc = np.zeros([len(node_idx), self.num_classes, num_steps + 1], dtype=np.float32)
		nlc[:, 0, :] = 1.0
		for node in labeled_nodes:
			n_idx = node_idx[node]
			nlc[n_idx, :, :] = 1.0

		self.propagate_seeds(labeled_nodes, unlabeled_nodes, graph, nls, nlc, node_labels, node_idx)

		for s_idx in range(1, num_steps + 1):
			self.propagate(unlabeled_nodes, graph, nls, nlc, s_idx, node_idx)

		nls = nls.mean(axis=-1)
		nls[:, 0] = nls[:, 1:].min()

		return nls, node_idx


class MultiHopLogConsistencyScoring(MultiHopConsistencyScoring):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def initialize(self, adj_list, seed_node_labels):
		g = nx.Graph()
		# list of (ex_t_id, ex_p_id, ex_tmp_energy)
		# 0 - entail
		# 1 - contradict
		nodes = set()
		unlabeled_nodes = []
		labeled_nodes = []
		node_idx = {}
		for t_id, p_id, tp_r_dists in adj_list:
			if t_id not in nodes:
				node_idx[t_id] = len(node_idx)
				if t_id in seed_node_labels:
					labeled_nodes.append(t_id)
				else:
					unlabeled_nodes.append(t_id)
			if p_id not in nodes:
				node_idx[p_id] = len(node_idx)
				if p_id in seed_node_labels:
					labeled_nodes.append(p_id)
				else:
					unlabeled_nodes.append(p_id)

			nodes.add(t_id)
			nodes.add(p_id)
			entail_weight, contradict_weight = tp_r_dists
			entail_weight = -np.log(entail_weight.item())
			contradict_weight = -np.log(contradict_weight.item())
			g.add_edge(t_id, p_id, entail_weight=entail_weight, contradict_weight=contradict_weight)
		return g, unlabeled_nodes, labeled_nodes, node_idx
