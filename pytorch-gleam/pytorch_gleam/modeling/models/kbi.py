
from collections import defaultdict
from functools import partial

import torch

from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds import ThresholdModule, MultiClassCallableThresholdModule
from pytorch_gleam.modeling.knowledge_embedding import KnowledgeEmbedding
from pytorch_gleam.modeling.metrics import Metric
from pytorch_gleam.inference import *


# noinspection PyAbstractClass
class KbiLanguageModel(BaseLanguageModel):
	def __init__(
			self,
			ke: KnowledgeEmbedding,
			infer: ConsistencyInference,
			threshold: ThresholdModule,
			metric: Metric,
			m_metric: Metric,
			num_relations: int = 2,
			num_classes: int = 3,
			num_val_seeds: int = 1,
			num_threshold_steps: int = 100,
			update_threshold: bool = True,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.num_relations = num_relations
		self.num_classes = num_classes
		self.ke = ke
		self.infer = infer
		self.num_val_seeds = num_val_seeds
		# TODO build multi-class multi-label threshold module
		self.threshold = threshold
		self.m_metric = m_metric
		self.num_threshold_steps = num_threshold_steps
		self.update_threshold = update_threshold

		self.ke_rel_layers = torch.nn.ModuleList(
			[
				torch.nn.Linear(
					in_features=self.hidden_size,
					out_features=self.ke.hidden_size
				) for _ in range(self.num_relations)
			]
		)
		self.ke_entity_layer = torch.nn.Linear(
			in_features=self.hidden_size,
			out_features=self.ke.hidden_size
		)
		self.f_dropout = torch.nn.Dropout(
			p=self.hidden_dropout_prob
		)

		self.metric = metric

	def eval_epoch_end(self, outputs, stage):
		triplet_eval_outputs, infer_eval_outputs = outputs
		loss = torch.cat([x['loss'] for x in triplet_eval_outputs], dim=0).mean()
		accuracy = torch.cat([x['accuracy'] for x in triplet_eval_outputs], dim=0).mean()
		self.log(f'{stage}_loss', loss)
		self.log(f'{stage}_accuracy', accuracy)

		self.threshold.cpu()
		# stage 0 is validation
		# stage 1 is test
		m_adj_list, m_labels, max_score, min_score = build_adj_list(infer_eval_outputs)
		m_s_labels, m_m_ids, m_t_ids = build_stage_labels(m_labels)

		num_val_seeds = self.num_val_seeds
		infer = self.infer
		for m_id in m_labels:
			if m_id not in self.threshold:
				self.threshold[m_id] = MultiClassCallableThresholdModule()

		def m_predict(m_id, m_threshold):
			m_i_adj = m_adj_list[m_id]
			m_s_t_labels = m_labels[m_id]
			# always use stage 0 (val) for seeds
			m_t_labels = m_s_t_labels[0]
			m_t_rel_labels = [(m_t_id, m_t_label) for (m_t_id, m_t_label) in m_t_labels.items() if m_t_label != 0]
			if stage == 'val':
				m_t_rel_labels = m_t_rel_labels[:num_val_seeds]
				m_i_adj = [
					(u_id, v_id, uv_scores) for (u_id, v_id, uv_scores) in m_i_adj
					if u_id in m_t_labels and v_id in m_t_labels
				]
			m_t_rel_labels = {m_t_id: m_t_label for (m_t_id, m_t_label) in m_t_rel_labels}

			if len(m_i_adj) == 0:
				m_s_i_preds = m_t_rel_labels
			else:
				m_s_i_preds = infer(m_i_adj, m_threshold, m_t_rel_labels)
			if stage != 'val':
				m_t_labels = m_s_t_labels[1]
			m_preds = []
			for ex_id, ex_l in m_t_labels.items():
				ex_pred = m_s_i_preds[ex_id]
				m_preds.append(ex_pred)
			m_preds = torch.tensor(m_preds, dtype=torch.long)

			return m_preds

		num_threshold_steps = self.num_threshold_steps
		m_metric = self.m_metric
		update_threshold = self.update_threshold
		threshold = self.threshold

		def infer_predict():
			preds = []
			for m_id, m_s_t_labels in m_labels.items():
				m_threshold = threshold[m_id]
				m_ls = []
				if stage != 'val':
					m_t_labels = m_s_t_labels[1]
				else:
					m_t_labels = m_s_t_labels[0]
				for ex_id, ex_l in m_t_labels.items():
					m_ls.append(ex_l)
				m_ls = torch.tensor(m_ls, dtype=torch.long)
				m_a_list = m_adj_list[m_id]
				if len(m_a_list) == 0:
					continue
				if update_threshold:
					m_scores = torch.cat([-dists for (_, _, dists) in m_a_list], dim=0)
					m_min_score = torch.min(m_scores).item()
					m_max_score = torch.max(m_scores).item()
					# check 100 values between min and max
					m_delta = (m_max_score - m_min_score) / num_threshold_steps

					max_threshold, max_metrics = m_metric.best(
						m_ls,
						partial(m_predict, m_id),
						m_threshold,
						threshold_min=m_min_score,
						threshold_max=m_max_score,
						threshold_delta=m_delta,
					)
					m_threshold.update_thresholds(max_threshold)

				m_t = m_threshold.thresholds.item()
				m_preds = m_predict(m_id, m_t)
				preds.append(m_preds)

			preds = torch.cat(preds, dim=0)
			return preds

		if stage == 'val':
			m_s_labels = m_s_labels[0]
		else:
			m_s_labels = m_s_labels[1]
		m_s_labels = torch.tensor(m_s_labels, dtype=torch.long)

		m_s_preds = infer_predict()
		f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(
			m_s_labels,
			m_s_preds
		)
		self.log(f'{stage}_loss', loss)
		self.log(f'{stage}_f1', f1)
		self.log(f'{stage}_p', p)
		self.log(f'{stage}_r', r)
		for t_idx, threshold in enumerate(self.threshold.thresholds):
			self.log(f'{stage}_threshold_{t_idx}', threshold)
		for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
			self.log(f'{stage}_{cls_index}_f1', c_f1)
			self.log(f'{stage}_{cls_index}_p', c_p)
			self.log(f'{stage}_{cls_index}_r', c_r)

		self.threshold.to(self.device)

	def eval_step(self, batch, batch_idx, dataloader_idx=None):
		if dataloader_idx is None or dataloader_idx == 0:
			loss, accuracy = self.triplet_step(batch)
			result = {
				'loss': loss,
				'accuracy': accuracy,
			}
		else:
			result = self.predict_step(batch, batch_idx, dataloader_idx)

		return result

	def forward(self, batch):
		num_examples = batch['num_examples']
		num_sequences_per_example = batch['num_sequences_per_example']
		num_entities = num_sequences_per_example - 1
		pad_seq_len = batch['pad_seq_len']

		# [bsize, num_seq, seq_len] -> [bsize * num_seq, seq_len]
		input_ids = batch['input_ids'].view(num_examples * num_sequences_per_example, pad_seq_len)
		attention_mask = batch['attention_mask'].view(num_examples * num_sequences_per_example, pad_seq_len)
		if 'token_type_ids' in batch:
			token_type_ids = batch['token_type_ids'].view(num_examples * num_sequences_per_example, pad_seq_len)
		else:
			token_type_ids = None
		# [bsize * num_seq, seq_len, hidden_size]
		contextualized_embeddings = self.lm_step(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		# [bsize * num_seq, hidden_size]
		lm_output = contextualized_embeddings[:, 0]
		# TODO consider dropout
		lm_output = self.f_dropout(lm_output)
		lm_output = lm_output.view(num_examples, num_sequences_per_example, self.hidden_size)
		# [bsize, hidden_size]
		r_lm_output = lm_output[:, 0]
		# [bsize * num_entities, hidden_size]
		e_lm_output = lm_output[:, 1:].reshape(num_examples * num_entities, self.hidden_size)
		e_proj = self.ke_entity_layer(e_lm_output)
		e_embs = self.ke(e_proj, 'entity')
		# [bsize, num_entities, emb_size]
		e_embs = e_embs.view(num_examples, num_entities, e_embs.shape[-1])
		# num_samples = batch['pos_samples'] + batch['neg_samples']
		# [bsize, num_samples, num_relations]
		# relation_mask = batch['relation_mask']
		r_projections = []
		for r_layer in self.ke_rel_layers:
			r_proj = r_layer(r_lm_output)
			r_projections.append(r_proj)
		# [bsize, num_relations, ke_hidden_size]
		r_projections = torch.stack(r_projections, dim=1)
		# [bsize * num_relations, ke_hidden_size]
		r_projections = r_projections.view(num_examples * self.num_relations, r_projections.shape[-1])
		# [bsize * num_relations, ke_emb_size]
		r_embs = self.ke(r_projections, 'rel')
		# [bsize, num_relations, ke_emb_size]
		r_embs = r_embs.view(num_examples, self.num_relations, r_embs.shape[-1])
		# [bsize, num_entities, emb_size]
		return e_embs, r_embs

	@staticmethod
	def split_embeddings(embs, batch):
		t_ex_embs = embs[:, 0]
		pos_samples = batch['pos_samples']
		if pos_samples > 0:
			pos_embs = embs[:, 1:1+pos_samples]
		else:
			pos_embs = None
		neg_samples = batch['neg_samples']
		if neg_samples > 0:
			neg_embs = embs[:, 1+pos_samples:1+pos_samples+neg_samples]
		else:
			neg_embs = None
		return t_ex_embs, pos_embs, neg_embs

	def predict_energy(self, e_embs, m_embs, batch):
		# convert m_embs to both relation types
		# [bsize, 1, num_relations, emb_size]
		m_embs = m_embs.unsqueeze(dim=-3)
		# [bsize, pos_samples, num_relations, 1]
		rel_mask = batch['relation_mask'].unsqueeze(dim=-1)
		# [bsize, pos_samples, num_relations, emb_size]
		m_embs = m_embs * rel_mask

		# t_ex_embs: [bsize, emb_size],
		# pos_embs: [bsize, pos_samples, emb_size]
		t_ex_embs, pos_embs, _ = self.split_embeddings(e_embs, batch)
		# [bsize, 1, 1, emb_size]
		t_ex_embs = t_ex_embs.unsqueeze(dim=-2).unsqueeze(dim=-2)
		# [bsize, pos_samples, 1, emb_size]
		pos_embs = pos_embs.unsqueeze(dim=-2)
		# [bsize, 1, 1, 2]
		d_mask = batch['direction_mask'].unsqueeze(dim=-2).unsqueeze(dim=-2)
		# [bsize, pos_samples, num_relations]
		pos_energy = self._energy(t_ex_embs, m_embs, pos_embs, d_mask)
		return pos_energy

	def triplet_energy(self, e_embs, m_embs, batch):
		# convert m_embs to only one relation type
		# [bsize, 1, num_relations, emb_size]
		m_embs = m_embs.unsqueeze(dim=-3)
		# [bsize, pos_samples + neg_samples, num_relations, 1]
		rel_mask = batch['relation_mask'].unsqueeze(dim=-1)
		# [bsize, pos_samples + neg_samples, num_relations, emb_size]
		m_embs = m_embs * rel_mask
		# using the rel_mask we are able to sum over m_embs that are zero'd
		m_embs = m_embs.sum(dim=-2)

		# t_ex_embs: [bsize, emb_size],
		# pos_embs: [bsize, pos_samples, emb_size],
		# neg_embs: [bsize, pos_samples, emb_size],
		t_ex_embs, pos_embs, neg_embs = self.split_embeddings(e_embs, batch)
		# [bsize, 1, emb_size]
		t_ex_embs = t_ex_embs.unsqueeze(dim=-2)
		pos_samples = batch['pos_samples']
		neg_samples = batch['neg_samples']
		# [bsize, pos_samples + neg_samples, emb_size]
		# [bsize, pos_samples, emb_size]
		pos_m_embs = m_embs[..., :pos_samples, :]
		# [bsize, neg_samples, emb_size]
		neg_m_embs = m_embs[..., pos_samples:pos_samples+neg_samples, :]

		# [bsize, 1, 2]
		d_mask = batch['direction_mask'].unsqueeze(dim=-2)
		# [bsize, pos_samples]
		pos_energy = self._energy(t_ex_embs, pos_m_embs, pos_embs, d_mask)
		# [bsize, neg_samples]
		neg_energy = self._energy(t_ex_embs, neg_m_embs, neg_embs, d_mask)

		return pos_energy, neg_energy

	def _energy(self, t_embs, m_embs, e_embs, direction_mask):
		# [bsize, pos_samples]
		forward_energy = self.ke.energy(t_embs, m_embs, e_embs)
		# [bsize, pos_samples]
		backward_energy = self.ke.energy(e_embs, m_embs, t_embs)
		# [bsize, pos_samples, 2]
		tme_energy = torch.stack([forward_energy, backward_energy], dim=-1)
		# [bsize, 1]
		d_sum = direction_mask.sum(dim=-1)
		# [bsize, pos_samples, 2] x [bsize, 1, 2] sum
		# -> [bsize, pos_samples] / [bsize, 1]
		# [bsize, pos_samples]
		tme_energy = (tme_energy * direction_mask).sum(dim=-1) / d_sum
		return tme_energy

	def loss(self, pos_energy, neg_energy):
		loss, accuracy = self.ke.loss(pos_energy, neg_energy)
		return loss, accuracy

	def triplet_step(self, batch):
		e_embs, r_embs = self(batch)
		pos_energy, neg_energy = self.triplet_energy(e_embs, r_embs, batch)
		loss, accuracy = self.loss(pos_energy, neg_energy)
		return loss, accuracy

	def training_step(self, batch, batch_idx):
		loss, accuracy = self.triplet_step(batch)
		accuracy = accuracy.mean()
		loss = loss.mean()
		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
		result = {
			'loss': loss
		}
		return result

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		e_embs, r_embs = self(batch)
		# [bsize, num_pairs, num_relations]
		pair_rel_energy = self.predict_energy(e_embs, r_embs, batch)
		results = {
			# [bsize]
			'ids': batch['ids'],
			# [bsize]
			'm_ids': batch['m_ids'],
			# [bsize, num_pairs]
			'p_ids': batch['p_ids'],
			# [bsize, num_pairs+1]
			'labels': batch['labels'],
			# [bsize, num_pairs+1]
			'stages': batch['stages'],
			# [bsize, num_pairs, num_relations]
			'energies': pair_rel_energy
		}
		return results


def flatten(l):
	return [item for sublist in l for item in sublist]


def build_adj_list(outputs):
	# [count]
	t_ids = flatten([x['ids'] for x in outputs])
	# [count]
	m_ids = flatten([x['m_ids'] for x in outputs])
	# [count]
	p_ids = flatten([x['p_ids'] for x in outputs])
	# [count, 2]
	labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu()
	stages = torch.cat([x['stages'] for x in outputs], dim=0).cpu()
	# [count]
	t_label = labels[:, 0]
	# [count]
	t_stage = stages[:, 0]
	# [count, 1]
	p_labels = labels[:, 1:]
	# [count, 1]
	p_stage = stages[:, 1:]

	# [count, 1, num_relations]
	t_energies = torch.cat([x['energies'] for x in outputs], dim=0).cpu()
	max_score = -torch.min(t_energies).item()
	min_score = -torch.max(t_energies).item()

	m_adj_list = defaultdict(list)
	m_labels = defaultdict(lambda: defaultdict(dict))
	for ex_idx in range(len(t_ids)):
		ex_t_id = t_ids[ex_idx]
		ex_m_id = m_ids[ex_idx]
		ex_p_ids = [p_ids[ex_idx]]
		ex_t_label = t_label[ex_idx]
		ex_t_stage = int(t_stage[ex_idx])
		ex_p_labels = p_labels[ex_idx]
		ex_p_stage = p_stage[ex_idx]
		ex_t_energies = t_energies[ex_idx]
		m_labels[ex_m_id][ex_t_stage][ex_t_id] = ex_t_label
		for p_idx in range(len(ex_p_ids)):
			ex_p_id = ex_p_ids[p_idx]
			ex_p_label = ex_p_labels[p_idx]
			ex_p_stage = int(ex_p_stage[p_idx])
			ex_tmp_energy = ex_t_energies[p_idx]
			m_adj_list[ex_m_id].append((ex_t_id, ex_p_id, ex_tmp_energy))
			m_labels[ex_m_id][ex_p_stage][ex_p_id] = ex_p_label

	return m_adj_list, m_labels, max_score, min_score


def build_stage_labels(m_labels):
	m_s_labels = defaultdict(list)
	m_m_ids = defaultdict(list)
	m_t_ids = defaultdict(list)
	for m_id, m_t_labels in m_labels.items():
		for stage_idx, stage_labels in m_t_labels.items():
			for m_t_id, m_t_label in stage_labels.items():
				m_t_ids[stage_idx].append(m_t_id)
				m_m_ids[stage_idx].append(m_id)
				m_s_labels[stage_idx].append(m_t_label)
	return m_s_labels, m_m_ids, m_t_ids
