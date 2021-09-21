
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from pytorch_gleam.data.datasets.kbi_misinfo_stance import flip_tm_stance, tmp_stance
from networkx.algorithms.traversal.breadth_first_search import bfs_predecessors
import heapq


def infer_clusters(
		adj_list: List[Tuple[str, str, Tuple[float, float]]],
		threshold: float,
		node_labels: Dict[str, int]
):
	seed_node_labels = node_labels.copy()
	node_labels = node_labels.copy()
	g = nx.Graph()
	# list of (ex_t_id, ex_p_id, ex_tmp_energy)
	# 0 - entail
	# 1 - contradict
	for t_id, p_id, tp_r_dists in adj_list:
		entail_weight, contradict_weight = tp_r_dists
		g.add_edge(t_id, p_id, entail_weight=entail_weight, contradict_weight=contradict_weight)

	node_relations = defaultdict(int)
	first_node = None
	for node in g.nodes():
		if node not in seed_node_labels:
			continue
		node_label = seed_node_labels[node]
		if node_label == 0:
			continue
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
			if max_label == 'entail' or max_label == 'contradict':
				node_relations[node] += 1

	max_rel_node, max_rel_count = max(node_relations.items(), key=lambda x: x[1], default=(first_node, 0))

	# assume the node with the most entailments is stance 1
	for node, _ in bfs_predecessors(g, max_rel_node):
		# No need to re-label labeled nodes
		if node in node_labels:
			continue
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


def infer_seed_clusters(
		adj_list: List[Tuple[str, str, Tuple[float, float]]],
		threshold: float,
		node_labels: Dict[str, int]
):
	seed_node_labels = node_labels.copy()
	node_labels = node_labels.copy()
	g = nx.Graph()
	# list of (ex_t_id, ex_p_id, ex_tmp_energy)
	# 0 - entail
	# 1 - contradict
	for t_id, p_id, tp_r_dists in adj_list:
		entail_weight, contradict_weight = tp_r_dists
		g.add_edge(t_id, p_id, entail_weight=entail_weight, contradict_weight=contradict_weight)

	node_scores = defaultdict(list)
	first_node = None
	for node in g.nodes():
		if node not in seed_node_labels:
			continue
		node_label = seed_node_labels[node]
		if node_label == 0:
			continue
		first_node = node
		for other_node in g.neighbors(node):
			edge = g.get_edge_data(node, other_node)
			entail_weight = edge['entail_weight']
			contradict_weight = edge['contradict_weight']
			max_score, max_label = max(
				(-entail_weight, 'entail'),
				(-contradict_weight, 'contradict'),
				key=lambda x: x[0]
			)
			node_scores[node].append(max_score)

	node_avg_scores = {}
	for node, scores in node_scores.items():
		node_avg_scores[node] = np.mean(scores)

	max_rel_node, max_avg_score = max(node_avg_scores.items(), key=lambda x: x[1], default=(first_node, 0.0))

	# assume the node with the most entailments is stance 1
	for node, _ in bfs_predecessors(g, max_rel_node):
		# No need to re-label labeled nodes
		if node in node_labels:
			continue
		node_label_scores = defaultdict(list)
		for other_node in g.neighbors(node):
			if other_node in node_labels:
				other_pred = node_labels[other_node]
				# entailment, contradiction, or neither does not mean anything if we know
				# the label of the other node
				if other_pred == 0:
					continue
				edge = g.get_edge_data(node, other_node)
				entail_score = -edge['entail_weight']
				entail_label = other_pred
				contradict_score = -edge['contradict_weight']
				contradict_label = flip_tm_stance(other_pred)
				node_label_scores[entail_label].append(entail_score)
				node_label_scores[contradict_label].append(contradict_score)

		label_avg_scores = {}
		for node_label, scores in node_label_scores.items():
			label_avg_scores[node_label] = np.mean(scores)
		label_avg_scores[0] = threshold
		max_label, max_score = max(
			label_avg_scores.items(),
			key=lambda x: x[1],
		)
		node_labels[node] = max_label

	return node_labels


def infer_seed_only_clusters(
		adj_list: List[Tuple[str, str, Tuple[float, float]]],
		threshold: float,
		node_labels: Dict[str, int]
):
	seed_node_labels = node_labels.copy()
	node_labels = node_labels.copy()
	g = nx.Graph()
	# list of (ex_t_id, ex_p_id, ex_tmp_energy)
	# 0 - entail
	# 1 - contradict
	for t_id, p_id, tp_r_dists in adj_list:
		entail_weight, contradict_weight = tp_r_dists
		g.add_edge(t_id, p_id, entail_weight=entail_weight, contradict_weight=contradict_weight)

	node_scores = defaultdict(list)
	first_node = None
	for node in g.nodes():
		if node not in seed_node_labels:
			continue
		node_label = seed_node_labels[node]
		if node_label == 0:
			continue
		first_node = node
		for other_node in g.neighbors(node):
			edge = g.get_edge_data(node, other_node)
			entail_weight = edge['entail_weight']
			contradict_weight = edge['contradict_weight']
			max_score, max_label = max(
				(-entail_weight, 'entail'),
				(-contradict_weight, 'contradict'),
				key=lambda x: x[0]
			)
			node_scores[node].append(max_score)

	node_avg_scores = {}
	for node, scores in node_scores.items():
		node_avg_scores[node] = np.mean(scores)

	max_rel_node, max_avg_score = max(node_avg_scores.items(), key=lambda x: x[1], default=(first_node, 0.0))

	# assume the node with the most entailments is stance 1
	for node, _ in bfs_predecessors(g, max_rel_node):
		# No need to re-label labeled nodes
		if node in seed_node_labels:
			continue
		node_label_scores = defaultdict(list)
		for other_node in g.neighbors(node):
			if other_node in seed_node_labels:
				other_pred = seed_node_labels[other_node]
				# entailment, contradiction, or neither does not mean anything if we know
				# the label of the other node
				if other_pred == 0:
					continue
				edge = g.get_edge_data(node, other_node)
				entail_score = -edge['entail_weight']
				entail_label = other_pred
				contradict_score = -edge['contradict_weight']
				contradict_label = flip_tm_stance(other_pred)
				node_label_scores[entail_label].append(entail_score)
				node_label_scores[contradict_label].append(contradict_score)

		label_avg_scores = {}
		for node_label, scores in node_label_scores.items():
			label_avg_scores[node_label] = np.mean(scores)
		label_avg_scores[0] = threshold
		max_label, max_score = max(
			label_avg_scores.items(),
			key=lambda x: x[1],
		)
		node_labels[node] = max_label

	return node_labels


def infer_seed_min_clusters(
		adj_list: List[Tuple[str, str, Tuple[float, float]]],
		threshold: float,
		node_labels: Dict[str, int]
):
	seed_node_labels = node_labels.copy()
	node_labels = node_labels.copy()
	g = nx.Graph()
	# list of (ex_t_id, ex_p_id, ex_tmp_energy)
	# 0 - entail
	# 1 - contradict
	for t_id, p_id, tp_r_dists in adj_list:
		entail_weight, contradict_weight = tp_r_dists
		g.add_edge(t_id, p_id, entail_weight=entail_weight, contradict_weight=contradict_weight)

	edge_heap = []
	for node in g.nodes():
		if node not in seed_node_labels:
			continue
		node_label = seed_node_labels[node]
		if node_label == 0:
			continue
		for other_node in g.neighbors(node):
			if other_node in seed_node_labels:
				continue
			edge = g.get_edge_data(node, other_node)
			entail_weight = edge['entail_weight']
			contradict_weight = edge['contradict_weight']
			min_weight, min_relation = min([(entail_weight, 0), (contradict_weight, 1)], key=lambda x: x[0])
			heapq.heappush(edge_heap, (min_weight, (node, other_node, min_relation)))

	while edge_heap:
		weight, (other_node, node, relation) = heapq.heappop(edge_heap)
		if node in node_labels:
			continue
		if -weight <= threshold:
			break
		if len(node_labels) == len(g.nodes()):
			break
		other_label = node_labels[other_node]
		node_label = tmp_stance(relation, other_label)
		node_labels[node] = node_label
		for next_node in g.neighbors(node):
			if next_node in node_labels:
				continue
			edge = g.get_edge_data(node, next_node)
			entail_weight = edge['entail_weight']
			contradict_weight = edge['contradict_weight']
			min_weight, min_relation = min([(entail_weight, 0), (contradict_weight, 1)], key=lambda x: x[0])
			heapq.heappush(edge_heap, (min_weight, (node, next_node, min_relation)))

	for node in g.nodes():
		if node not in node_labels:
			node_labels[node] = 0

	return node_labels

