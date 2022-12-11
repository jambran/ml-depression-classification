import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
)

from .depression_detection_config import DepressionDetectionConfig


class DepressionDetectionModel(PreTrainedModel):
    config_class = DepressionDetectionConfig

    def __init__(self,
                 config: DepressionDetectionConfig,
                 ):
        super().__init__(config)

        self.encoder = Encoder(config.model_name_or_path)
        self.config = config
        layers = [
            nn.Linear(self.config.hidden_size,
                      self.config.hidden_size)
            for _ in range(self.config.num_layers)
        ]

        self.hidden_layers = nn.Sequential(*layers)
        self.classification_layer = nn.Linear(self.config.hidden_size,
                                              self.config.num_classes)

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.FloatTensor,
                ):
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # get the CLS token, the full sentence representation, first
        # token in the sequence
        # (bs, hidden_size)
        sentence_representation = encoded.last_hidden_state[:, 0, :]

        # (bs, seq_len, hidden_size)
        final_hidden_layer = self.hidden_layers(sentence_representation)

        logits = self.classification_layer(final_hidden_layer)

        return logits

    def make_computational_graph(self, logger, device: str):
        """
        following https://learnopencv.com/tensorboard-with-pytorch-lightning/
        :return:
        """
        sample_input = {
            'input_ids': torch.tensor([[101, 2307, 3105, 999, 102, 0, 0, 0]],
                                      device=device),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]],
                                           device=device),
            'label': torch.tensor([0], device=device),
        }
        logger.experiment.add_graph(self,
                                    (sample_input['input_ids'],
                                     sample_input['attention_mask']),
                                    )


class Encoder(nn.Module):

    def __init__(self, model_name_or_path: str = "distilbert-base-cased"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
        )
        self.model = AutoModel.from_config(self.config)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)
