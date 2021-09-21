from abc import abstractmethod, ABC

import torch

from pytorch_gleam.modeling.base_thresholds import ThresholdModule


class Metric(torch.nn.Module, ABC):
	def __init__(self, mode: str = 'max'):
		super().__init__()
		self.mode = mode
		if self.mode == 'max':
			self.mode_reduce = max
		elif self.mode == 'min':
			self.mode_reduce = min
		else:
			raise ValueError(f'Unknown mode: {self.mode}')

	@abstractmethod
	def forward(self, labels, predictions):
		pass

	def best(self, labels, scores, threshold: ThresholdModule, threshold_min: float = None, threshold_max: float = None, threshold_delta: float = None):
		max_threshold, max_metrics = self.mode_reduce(
			self.best_iterator(labels, scores, threshold, threshold_min, threshold_max, threshold_delta),
			key=lambda x: x[1][0]
		)
		return max_threshold, max_metrics

	def best_iterator(self, labels, scores, threshold: ThresholdModule, threshold_min: float = None, threshold_max: float = None, threshold_delta: float = None):
		for threshold, preds in threshold.get_range_predictions(scores, threshold_min, threshold_max, threshold_delta):
			metrics = self(labels, preds)
			yield threshold, metrics

