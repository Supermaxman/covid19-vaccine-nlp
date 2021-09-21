from collections import defaultdict

import networkx as nx
import numpy as np
from pytorch_gleam.data.datasets.kbi_misinfo_stance import flip_tm_stance
from networkx.algorithms.traversal.breadth_first_search import bfs_predecessors


def infer_clusters(adj_list, threshold: float):

	g = nx.Graph()
	# list of (ex_t_id, ex_p_id, ex_tmp_energy)
	# 0 - entail
	# 1 - contradict
	for t_id, p_id, tp_r_dists in adj_list:
		entail_weight, contradict_weight = tp_r_dists
		g.add_edge(t_id, p_id, entail_weight=entail_weight, contradict_weight=contradict_weight)

	node_entailments = defaultdict(int)
	first_node = None
	for node in g.nodes():
		first_node = node
		for other_node in g.neighbors(node):
			edge = g.get_edge_data(node, other_node)
			entail_weight = edge['entail_weight']
			contradict_weight = edge['contradict_weight']
			max_score, max_label = max(
				(-entail_weight, 'entail'),
				(-contradict_weight, 'contradict'),
				(threshold, 'none'),
				key=lambda x: x[0]
			)
			if max_label == 'entail':
				node_entailments[node] += 1

	max_entail_node, max_entail_count = max(node_entailments.items(), key=lambda x: x[1], default=(first_node, 0))

	# assume the node with the most entailments is stance 1
	node_labels = {max_entail_node: 1}
	for node, _ in bfs_predecessors(g, max_entail_node):
		# TODO v2: use average of scores for prediction vs threshold
		# this uses average of predictions
		node_label_counts = defaultdict(int)
		for other_node in g.neighbors(node):
			if other_node in node_labels:
				other_pred = node_labels[other_node]
				# entailment, contradiction, or neither does not mean anything if we know
				# the label of the other node
				if other_pred == 0:
					continue
				edge = g.get_edge_data(node, other_node)
				entail_weight = edge['entail_weight']
				contradict_weight = edge['contradict_weight']
				max_score, max_label = max(
					(-entail_weight, 'entail'),
					(-contradict_weight, 'contradict'),
					(threshold, 'none'),
					key=lambda x: x[0]
				)
				node_pred = 0
				if max_label == 'entail':
					node_pred = other_pred
				elif max_label == 'contradict':
					# 0 -> 0
					# 1 -> 2
					# 2 -> 1
					node_pred = flip_tm_stance(other_pred)
				node_label_counts[node_pred] += 1
		node_label_scores = defaultdict(float)
		node_label_total_count = sum(node_label_counts.values())
		for node_label, node_count in node_label_counts.items():
			node_label_scores[node_label] = node_count / node_label_total_count

		# if there are no scores for a node then default to no stance
		# this would only happen if all the adjacent nodes were assigned 0
		max_label, max_score = max(node_label_scores.items(), key=lambda x: x[-1], default=(0, 0))
		node_labels[node] = max_label

	return node_labels

