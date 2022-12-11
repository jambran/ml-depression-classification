from dataclasses import dataclass
from typing import (
    Dict,
)

import torch
from pydantic import BaseModel


@dataclass
class DetectionOutput:
    logits: torch.Tensor  # (bs, num_labels)
    probabilities: torch.Tensor  # (bs, num_labels)


class DepressionPrediction(BaseModel):
    """
    slightly more user focused object. At inference time, the information
    we'll return
    """
    predicted_label: str
    probability: float
    label_to_probability: Dict[str, float]
