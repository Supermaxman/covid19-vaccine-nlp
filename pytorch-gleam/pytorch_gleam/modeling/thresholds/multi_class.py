import torch

from pytorch_gleam.modeling.thresholds.base_thresholds import ThresholdModule


class MultiClassThresholdModule(ThresholdModule):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_thresholds=1, **kwargs)

	def predict(self, scores, thresholds):
		# zero class is idx 0
		# pos classes is idx 1, ...
		pos_scores = scores[..., 1:]
		# filter out non-threshold classes
		pos_above = pos_scores.gt(thresholds).float()
		# [bsize, num_labels-1]
		pos_scores = pos_scores * pos_above
		# 1 if any are above threshold, 0 if none are above threshold
		# [bsize]
		pos_any_above = (pos_above.sum(dim=-1).gt(0)).long()
		# if none are above threshold then our prediction will be class 0, otherwise it will be
		# between the classes which have scores above the threshold
		# [bsize]
		# we add one to the class id to account for the [:, 1:] filtering of only positive scores
		pos_predictions = (pos_scores.max(dim=-1)[1] + 1)
		# [bsize]
		predictions = pos_predictions * pos_any_above
		return predictions


class MultiClassCallableThresholdModule(MultiClassThresholdModule):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def predict(self, scores, thresholds):
		predictions = scores(thresholds)
		return predictions

