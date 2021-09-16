
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
		self.threshold_range = torch.arange(
			self.threshold_min,
			self.threshold_max,
			self.threshold_delta,
			dtype=torch.float32,
			requires_grad=False
		)

	def forward(self, scores):
		return self.predict(scores, self.thresholds)

	def predict(self, scores, thresholds):
		preds = scores.gt(thresholds).long()
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

