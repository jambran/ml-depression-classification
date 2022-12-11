from operator import itemgetter
from typing import List

import torch.nn.functional as F
from transformers.pipelines import TextClassificationPipeline

from .data_loader import DepressionLabel
from .data_models import DepressionPrediction


def make_prediction_object(probabilities: List[float],
                           ) -> DepressionPrediction:
    labels_and_probs = list(zip(DepressionLabel, probabilities))
    best_label, best_prob = max(labels_and_probs, key=itemgetter(1))

    prediction = DepressionPrediction(
        predicted_label=best_label,
        probability=best_prob,
        label_to_probability={
            label.name: prob
            for label, prob in labels_and_probs
        },
    )
    return prediction


class DepressionPipeline(TextClassificationPipeline):

    def postprocess(self, logits, **kwargs):
        # first index since only one sentence at a time
        probabilities = F.softmax(logits, dim=1).tolist()

        predictions = [
            make_prediction_object(classwise_probs)
            for classwise_probs in probabilities
        ]
        return predictions
