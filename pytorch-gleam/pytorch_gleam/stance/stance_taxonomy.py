

from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
import os
from multiprocessing import Pool
from typing import List

import ujson as json
from tqdm import tqdm
import numpy as np
import pandas as pd


class Taxonomy(ABC):
	def __init__(self, name: str):
		self.name = name
		self.themes = None
		self.frames = None

	@abstractmethod
	def load_themes(self, theme_path):
		pass

	@abstractmethod
	def theme_score(self, scores, frames):
		pass

	@abstractmethod
	def frames_to_themes(self):
		pass


class HierarchicalTaxonomy(Taxonomy, ABC):
	def __init__(self, name: str):
		super().__init__(name=name)
		self.concerns = None

	@abstractmethod
	def load_concerns(self, file_path):
		pass


class MisinformationTaxonomy(HierarchicalTaxonomy):
	def __init__(self, concern_path, theme_path, frames):
		super().__init__(name='misinformation')
		self.concerns = self.load_concerns(concern_path)
		self.themes = self.load_themes(theme_path)
		accept_mask = frames['misinformation|Accept'].apply(len) > 0
		reject_mask = frames['misinformation|Reject'].apply(len) > 0
		self.frames = frames[accept_mask | reject_mask]

	def load_concerns(self, file_path):
		m_concerns = pd.read_excel(
			file_path,
			index_col='m_id',
			engine='openpyxl',
			dtype=str
		)
		m_concerns.index = m_concerns.index.astype(str)
		return m_concerns

	def load_themes(self, file_path):
		m_themes = pd.read_excel(
			file_path,
			index_col='theme_id',
			engine='openpyxl',
			dtype=str
		)
		m_themes.index = m_themes.index.astype(str)
		return m_themes

	def frames_to_themes(self):
		f_lookup = defaultdict(set)
		for f_id, frame in self.frames.iterrows():
			for m_id in frame['misinformation|Accept']:
				if len(m_id) > 0:
					mt_id = self.concerns.loc[m_id]['theme']
					# ensure same theme only gets score once for one frame - stance pair, no need to count multiple times
					f_lookup[f_id].add((mt_id, 1))
			for m_id in frame['misinformation|Reject']:
				if len(m_id) > 0:
					mt_id = self.concerns.loc[m_id]['theme']
					# ensure same theme only gets score once for one frame - stance pair, no need to count multiple times
					f_lookup[f_id].add((mt_id, -1))
		return f_lookup

	def theme_score(self, scores, frames):
		mt_scores = defaultdict(list)
		for fs_score, f_id in scores:
			fs_themes = {}
			for m_id in frames.loc[f_id]['misinformation|Accept']:
				if len(m_id) > 0:
					mt_id = self.concerns.loc[m_id]['theme']
					# ensure same theme only gets score once for one frame - stance pair, no need to count multiple times
					fs_themes[mt_id] = fs_score
			for m_id in frames.loc[f_id]['misinformation|Reject']:
				if len(m_id) > 0:
					mt_id = self.concerns.loc[m_id]['theme']
					# ensure same theme only gets score once for one frame - stance pair, no need to count multiple times
					fs_themes[mt_id] = -fs_score
			for mt_id, mt_score in fs_themes.items():
				mt_scores[mt_id].append(mt_score)

		themes = []
		for mt_id, mt_i_scores in mt_scores.items():
			mt_score = np.sum(mt_i_scores)
			themes.append((mt_score, mt_id))
		themes = list(sorted(themes, key=lambda x: -np.abs(x[0])))
		return themes


class TrustTaxonomy(HierarchicalTaxonomy):
	def __init__(self, name, concern_path, theme_path, frames):
		super().__init__(name=name)
		self.concerns = self.load_concerns(concern_path)
		self.themes = self.load_themes(theme_path)
		accept_mask = frames['trust|Accept'].apply(len) > 0
		reject_mask = frames['trust|Reject'].apply(len) > 0
		self.frames = frames[accept_mask | reject_mask]

	def load_concerns(self, file_path):
		t_concerns = pd.read_excel(
			file_path,
			index_col='trust_id',
			engine='openpyxl',
			dtype=str
		).fillna('')
		t_concerns.index = t_concerns.index.astype(str)
		return t_concerns

	def load_themes(self, file_path):
		t_themes = pd.read_excel(
			file_path,
			index_col='theme_id',
			engine='openpyxl',
			dtype=str
		)
		t_themes.index = t_themes.index.astype(str)
		return t_themes

	def frames_to_themes(self):
		f_lookup = defaultdict(set)
		for f_id, frame in self.frames.iterrows():
			for t_id in frame['trust|Accept']:
				if len(t_id) > 0 and t_id in self.concerns.index:
					# ensure same theme only gets score once for one frame - stance pair, no need to count multiple times
					tp_id = self.concerns.loc[t_id]['theme']
					if len(tp_id) > 0:
						f_lookup[f_id].add((tp_id, 1))
			for t_id in frame['trust|Reject']:
				if len(t_id) > 0 and t_id in self.concerns.index:
					# ensure same theme only gets score once for one frame - stance pair, no need to count multiple times
					tp_id = self.concerns.loc[t_id]['theme']
					if len(tp_id) > 0:
						f_lookup[f_id].add((tp_id, -1))
		return f_lookup

	def theme_score(self, scores, frames):
		tp_scores = defaultdict(list)
		for fs_score, f_id in scores:
			fs_tp_themes = {}
			for t_id in frames.loc[f_id]['trust|Accept']:
				if len(t_id) > 0 and t_id in self.concerns.index:
					# ensure same theme only gets score once for one frame - stance pair, no need to count multiple times
					t_score = fs_score
					tp_id = self.concerns.loc[t_id]['theme']
					if len(tp_id) > 0:
						fs_tp_themes[tp_id] = t_score
			for t_id in frames.loc[f_id]['trust|Reject']:
				if len(t_id) > 0 and t_id in self.concerns.index:
					# ensure same theme only gets score once for one frame - stance pair, no need to count multiple times
					t_score = -fs_score
					tp_id = self.concerns.loc[t_id]['theme']
					if len(tp_id) > 0:
						fs_tp_themes[tp_id] = t_score
			for tp_id, t_score in fs_tp_themes.items():
				tp_scores[tp_id].append(t_score)
		p_themes = []
		for t_id, t_i_scores in tp_scores.items():
			t_score = np.sum(t_i_scores)
			p_themes.append((t_score, t_id))
		p_themes = list(sorted(p_themes, key=lambda x: -np.abs(x[0])))

		return p_themes


class TrustBuildingTaxonomy(TrustTaxonomy):
	def __init__(self, concern_path, theme_path, frames):
		super().__init__(name='trust+', concern_path=concern_path, theme_path=theme_path, frames=frames)


class TrustErodingTaxonomy(TrustTaxonomy):
	def __init__(self, concern_path, theme_path, frames):
		super().__init__(name='trust-', concern_path=concern_path, theme_path=theme_path, frames=frames)


class MoralityTaxonomy(Taxonomy):
	def __init__(self, theme_path, frames):
		super().__init__(name='morality')
		self.themes = self.load_themes(theme_path)
		# all frames have moralities defined
		self.frames = frames

	def load_themes(self, file_path):
		mo = [
			'Care', 'Harm',
			'Fairness', 'Cheating',
			'Loyalty', 'Betrayal',
			'Authority', 'Subversion',
			'Purity', 'Degradation'
		]
		mo = pd.DataFrame(
			columns=['theme_id', 'text'],
			data=[(x, x) for x in mo])
		return mo

	def frames_to_themes(self):
		f_lookup = defaultdict(set)
		for f_id, frame in self.frames.iterrows():
			for m_id in frame['morality']:
				if len(m_id) > 0:
					f_lookup[f_id].add((m_id, 1))
		return f_lookup

	def theme_score(self, scores, frames):
		m_scores = defaultdict(list)
		for fs_score, f_id in scores:
			for m_id in frames.loc[f_id]['morality']:
				if len(m_id) > 0:
					m_scores[m_id].append(fs_score)
		misinfo = []
		for m_id, m_i_scores in m_scores.items():
			m_score = np.sum(m_i_scores)
			misinfo.append((m_score, m_id))
		misinfo = list(sorted(misinfo, key=lambda x: -np.abs(x[0])))
		return misinfo


class LiteracyTaxonomy(Taxonomy):
	def __init__(self, theme_path, frames):
		super().__init__(name='literacy')
		self.themes = self.load_themes(theme_path)

		accept_mask = frames['literacy+|Accept'].apply(len) > 0
		reject_mask = frames['literacy-|Accept'].apply(len) > 0
		self.frames = frames[accept_mask | reject_mask]

	def load_themes(self, file_path):
		lt_themes = pd.DataFrame(columns=['theme_id', 'text'], data=[['+', '+'], ['-', '-']])
		return lt_themes

	def frames_to_themes(self):
		f_lookup = defaultdict(set)
		for f_id, frame in self.frames.iterrows():
			for m_id in frame['literacy+|Accept']:
				if len(m_id) > 0:
					f_lookup[f_id].add(('+', 1))
			for m_id in frame['literacy-|Accept']:
				if len(m_id) > 0:
					f_lookup[f_id].add(('-', 1))
		return f_lookup

	def theme_score(self, scores, frames):
		m_scores = defaultdict(list)
		for fs_score, f_id in scores:
			for m_id in frames.loc[f_id]['literacy+|Accept']:
				if len(m_id) > 0:
					m_scores['+'].append(fs_score)
					break
			for m_id in frames.loc[f_id]['literacy-|Accept']:
				if len(m_id) > 0:
					m_scores['-'].append(fs_score)
					break
		misinfo = []
		for m_id, m_i_scores in m_scores.items():
			m_score = np.sum(m_i_scores)
			misinfo.append((m_score, m_id))
		misinfo = list(sorted(misinfo, key=lambda x: -np.abs(x[0])))
		return misinfo


class CivilRightsTaxonomy(Taxonomy):
	def __init__(self, theme_path, frames):
		super().__init__(name='civil_rights')
		self.themes = self.load_themes(theme_path)

		accept_mask = frames['civil_rights+|Accept'].apply(len) > 0
		reject_mask = frames['civil_rights-|Accept'].apply(len) > 0
		self.frames = frames[accept_mask | reject_mask]

	def load_themes(self, file_path):
		cr_themes = pd.DataFrame(columns=['theme_id', 'text'], data=[['+', '+'], ['-', '-']])
		return cr_themes

	def frames_to_themes(self):
		f_lookup = defaultdict(set)
		for f_id, frame in self.frames.iterrows():
			for m_id in frame['civil_rights+|Accept']:
				if len(m_id) > 0:
					f_lookup[f_id].add(('+', 1))
			for m_id in frame['civil_rights-|Accept']:
				if len(m_id) > 0:
					f_lookup[f_id].add(('-', 1))
		return f_lookup

	def theme_score(self, scores, frames):
		m_scores = defaultdict(list)
		for fs_score, f_id in scores:
			for m_id in frames.loc[f_id]['civil_rights+|Accept']:
				if len(m_id) > 0:
					m_scores['+'].append(fs_score)
					break
			for m_id in frames.loc[f_id]['civil_rights-|Accept']:
				if len(m_id) > 0:
					m_scores['-'].append(fs_score)
					break
		misinfo = []
		for m_id, m_i_scores in m_scores.items():
			m_score = np.sum(m_i_scores)
			misinfo.append((m_score, m_id))
		misinfo = list(sorted(misinfo, key=lambda x: -np.abs(x[0])))
		return misinfo


def split_taxonomy(x):
	return [str(int(float(x))) for x in str(x).split(',') if len(x) > 0]


def split_morality(x):
	return str(x).split()


class FrameTaxonomy(object):
	def __init__(self, frames, taxonomies: List[Taxonomy]):
		super().__init__()
		self.frames = frames
		self.taxonomies = taxonomies
		themes = []
		for tax in self.taxonomies:
			tax_theme = tax.themes
			tax_theme = tax_theme.reset_index()
			tax_theme['taxonomy'] = tax.name
			themes.append(tax_theme)
		themes = pd.concat(themes)
		themes['idx'] = range(len(themes))
		self.themes = themes.set_index(['taxonomy', 'theme_id'])
		self.themes_inv = themes.reset_index().set_index('idx')
		self.f_map = {f_id: idx for (idx, f_id) in enumerate(self.frames.index)}
		self.f_imap = {v: k for (k, v) in self.f_map.items()}

	def frames_to_themes(self):
		f_lookup = defaultdict(list)
		for tax in self.taxonomies:
			tax_f_lookup = tax.frames_to_themes()
			for f_id, tax_set in tax_f_lookup.items():
				for tax_theme_id, score_sign in tax_set:
					theme_idx = self.themes.loc[tax.name, tax_theme_id]['idx']
					f_lookup[f_id].append((int(theme_idx), score_sign))
		return f_lookup

	@staticmethod
	def load_frames(file_path: str):
		taxonomy_cols = [
			'misinformation|Accept',
			'misinformation|Reject',
			'trust|Accept',
			'trust|Reject',
			'civil_rights+|Accept',
			'civil_rights-|Accept',
			'literacy+|Accept',
			'literacy-|Accept',
		]

		frames = pd.read_excel(
			file_path,
			index_col='f_id',
			engine='openpyxl'
		)
		for col in taxonomy_cols:
			frames[col] = frames[col].fillna('').apply(split_taxonomy)

		# morality
		frames['morality'] = frames['morality'].apply(split_morality)
		return frames

	def frame_stance_score(self, v, threshold=0.2):
		scores = []
		for i in np.argsort(np.abs(v))[::-1]:
			i_score = v[i]
			if np.abs(i_score) < threshold:
				break
			f_idx = i
			f_id = self.f_imap[f_idx]
			scores.append((i_score, f_id))
		return scores

	def theme_embedding(self, u, threshold=0.2):
		t = np.zeros([len(self.themes)], dtype=np.float32)
		stance_scores = self.frame_stance_score(u, threshold)
		for tax in self.taxonomies:
			tax_scores = tax.theme_score(stance_scores, self.frames)
			self.assign_score(t, tax_scores, tax.name)
		return t

	def theme_scores(self, t_vec, t_threshold=0.02):
		scores = []
		for t_idx in np.argsort(np.abs(t_vec))[::-1]:
			i_score = t_vec[t_idx]
			if np.abs(i_score) < t_threshold:
				break
			t = self.themes_inv.loc[t_idx]
			t_id = t['theme_id']
			t_tax = t['taxonomy']
			scores.append((i_score, t_tax, t_id))
		return scores

	def frame_embedding(self, tweet, threshold=0.2):
		t_vec = np.zeros([len(self.frames)], dtype=np.float32)
		set_value = False
		for f_id, f_s in tweet['stance'].items():
			f_idx = self.f_map[f_id]
			stance, score = max(f_s.items(), key=lambda x: x[1])
			if stance == 'No Stance' or score < threshold:
				continue
			if stance == 'Reject':
				score *= -1
			t_vec[f_idx] = score
			set_value = True
		if not set_value:
			return None
		return t_vec

	def assign_score(self, t, scores, tax):
		for t_score, t_id in scores:
			t_idx = self.themes.loc[tax, t_id]['idx']
			t[t_idx] = t_score


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-f', '--frame_map_output_path', required=True)
	parser.add_argument('-o', '--theme_output_path', required=True)
	args = parser.parse_args()

	# taxonomy_path = 'data/co-vax-frames/taxonomy'
	taxonomy_path = args.input_path
	frame_map_output_path = args.frame_map_output_path
	theme_output_path = args.theme_output_path

	frame_path = os.path.join(taxonomy_path, 'frames-covid19.xlsx')
	frames = FrameTaxonomy.load_frames(frame_path)
	tax = FrameTaxonomy(
		frames=frames,
		taxonomies=[
			MisinformationTaxonomy(
				concern_path=os.path.join(taxonomy_path, 'misinformation-concerns-covid19.xlsx'),
				theme_path=os.path.join(taxonomy_path, 'misinformation-themes-covid19.xlsx'),
				frames=frames
			),
			TrustBuildingTaxonomy(
				concern_path=os.path.join(taxonomy_path, 'trust-building-concerns-covid19.xlsx'),
				theme_path=os.path.join(taxonomy_path, 'trust-building-themes-covid19.xlsx'),
				frames=frames
			),
			TrustErodingTaxonomy(
				concern_path=os.path.join(taxonomy_path, 'trust-eroding-concerns-covid19.xlsx'),
				theme_path=os.path.join(taxonomy_path, 'trust-eroding-themes-covid19.xlsx'),
				frames=frames
			),
			LiteracyTaxonomy(
				theme_path=os.path.join(taxonomy_path, 'literacy-themes-covid19.xlsx'),
				frames=frames
			),
			CivilRightsTaxonomy(
				theme_path=os.path.join(taxonomy_path, 'civil-rights-themes-covid19.xlsx'),
				frames=frames
			),
			# MoralityTaxonomy(
			# 	theme_path=os.path.join(taxonomy_path, 'morality-themes-covid19.xlsx'),
			# 	frames=frames
			# ),
		]
	)
	f2t = tax.frames_to_themes()
	for f_id, f_list in f2t.items():
		print(f_id)
		print(f_list)
		print()
	t2idx = tax.themes
	with open(frame_map_output_path, 'w') as f:
		json.dump(f2t, f)

	with open(theme_output_path, 'w') as f:
		t2idx.to_json(f)


if __name__ == '__main__':
	main()
