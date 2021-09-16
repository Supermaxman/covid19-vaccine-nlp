import numpy as np

import torch


class ThresholdModule(torch.nn.Module):
	def __init__(
			self, num_thresholds: int, threshold_min: int = 0.0, threshold_max: int = 1.0, threshold_delta: int = 0.01):
		super().__init__()
		self.num_thresholds = num_thresholds
		self.threshold_min = threshold_min
		self.threshold_max = threshold_max
		self.threshold_delta = threshold_delta

		self.thresholds = torch.nn.Parameter(
			torch.zeros(num_thresholds, dtype=torch.float32),
			requires_grad=False
		)
		self.threshold_range = np.arange(
			self.threshold_min,
			self.threshold_max,
			self.threshold_delta,
			dtype=np.float32,
		)

	def forward(self, scores):
		return self.predict(scores, self.thresholds.numpy())

	def predict(self, scores: np.ndarray, thresholds):
		preds = (scores > thresholds).astype(np.int64)
		return preds

	def update_thresholds(self, new_value):
		self.thresholds.copy_(new_value)

	def get_range(self):
		return self.threshold_range

	def get_range_predictions(self, scores):
		if self.num_thresholds == 1:
			t_preds = self.get_range_threshold_predictions(scores)
			for threshold, preds in t_preds:
				yield threshold, preds
		else:
			raise NotImplementedError('Still need to implement this properly')
			# for t_idx in range(self.num_thresholds):
			# 	t_scores = scores[..., t_idx]
			# 	t_range_preds = self.get_range_threshold_predictions(t_scores)
			# 	for t_threshold, t_preds in t_range_preds:
			# 		yield t_threshold, t_preds

	def get_range_threshold_predictions(self, scores):
		for threshold in self.get_range():
			preds = self.predict(scores, threshold)
			yield threshold, preds


class MultiClassThresholdModule(ThresholdModule):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_thresholds=1, **kwargs)

	def predict(self, scores: np.ndarray, thresholds: np.ndarray):
		# zero class is idx 0
		# pos classes is idx 1, ...
		pos_scores = scores[..., 1:]
		# filter out non-threshold classes
		pos_above = (pos_scores > thresholds).astype(np.float32)
		# [bsize, num_labels-1]
		pos_scores = pos_scores * pos_above
		# 1 if any are above threshold, 0 if none are above threshold
		# [bsize]

		pos_any_above = (np.sum(pos_above, axis=-1) > 0).astype(np.int64)
		# if none are above threshold then our prediction will be class 0, otherwise it will be
		# between the classes which have scores above the threshold
		# [bsize]
		# we add one to the class id to account for the [:, 1:] filtering of only positive scores

		pos_predictions = (np.max(pos_scores, axis=-1)[1] + 1)
		# [bsize]
		predictions = pos_predictions * pos_any_above
		return predictions
