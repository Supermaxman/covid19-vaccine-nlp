
from pytorch_gleam.modeling.models import BaseLanguageModelForSequenceClassification


# noinspection PyAbstractClass
class ReRankLanguageModel(BaseLanguageModelForSequenceClassification):
	def __init__(
			self,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)

	def forward(self, batch):
		input_ids = batch['input_ids']
		attention_mask = batch['attention_mask']
		if 'token_type_ids' in batch:
			token_type_ids = batch['token_type_ids']
		else:
			token_type_ids = None
		logits = self.lm_step(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)

		return logits

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		logits = self(batch)
		results = {
			'ids': batch['ids'],
			'logits': logits
		}
		return results
