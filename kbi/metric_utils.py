
import torch
from threshold_utils import ThresholdModule
from sklearn.metrics import precision_recall_fscore_support


class Metric(torch.nn.Module):
	def __init__(self, mode: str = 'max'):
		super().__init__()
		self.mode = mode
		if self.mode == 'max':
			self.mode_reduce = max
		elif self.mode == 'min':
			self.mode_reduce = min
		else:
			raise ValueError(f'Unknown mode: {self.mode}')

	def forward(self, labels, predictions):
		pass

	def best(self, labels, scores, threshold: ThresholdModule):
		max_threshold, max_metrics = self.mode_reduce(
			self.best_iterator(labels, scores, threshold),
			key=lambda x: x[1][0]
		)
		return max_threshold, max_metrics

	def best_iterator(self, labels, scores, threshold: ThresholdModule):
		for threshold, preds in threshold.get_range_predictions(scores):
			metrics = self(labels, preds)
			yield threshold, metrics


class F1PRMultiClassMetric(Metric):
	def __init__(self, num_classes: int, mode: str = 'macro', *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.num_classes = num_classes
		self.mode = mode
		self.pos_labels = list(range(self.num_classes))[1:]

	def forward(self, labels, predictions):
		labels = labels.cpu().numpy()
		predictions = predictions.cpu().numpy()
		cls_precision, cls_recall, cls_f1, cls_sup = precision_recall_fscore_support(
			labels,
			predictions,
			average=None,
			labels=self.pos_labels,
			zero_division=0
		)
		if self.mode == 'macro':
			# ignore 0 class in macro average
			mode_f1 = cls_f1.mean()
			mode_precision = cls_precision.mean()
			mode_recall = cls_recall.mean()
		elif self.mode == 'micro':
			mode_precision, mode_recall, mode_f1, mode_sup = precision_recall_fscore_support(
				labels,
				predictions,
				average='micro',
				labels=self.pos_labels,
				zero_division=0
			)
		else:
			raise ValueError(f'Unknown metric mode: {self.mode}')

		return mode_f1, mode_precision, mode_recall, cls_f1, cls_precision, cls_recall, self.pos_labels
