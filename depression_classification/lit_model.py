from dataclasses import dataclass
from operator import itemgetter
from typing import Iterable, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from transformers import (
    AutoModel,
    AutoConfig,
)
from transformers import (
    AutoTokenizer,
)
from pydantic import BaseModel

from .data_loader import DepressionLabel
from .model import DepressionDetectionModel
from .plotting_metrics import (
    plot_prf1_per_class,
    plot_confusion_matrix,
)


@dataclass
class DetectionOutput:
    logits: torch.Tensor  # (bs, num_labels)
    probabilities: torch.Tensor  # (bs, num_labels)


class DepressionPrediction(BaseModel):
    """
    slightly more user focused object. At inference time, the information
    we'll return
    """
    sentence: str
    predicted_label: str
    probability: float
    label_to_probability: Dict[str, float]


class LitDepressionDetectionModel(pl.LightningModule):

    def __init__(self,
                 experiment_name: str,
                 model_name_or_path: str,
                 learning_rate: float,
                 dropout,
                 attention_dropout,
                 num_layers,
                 hidden_size,
                 num_classes,
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            dropout=self.hparams.dropout,
            attention_dropout=self.hparams.attention_dropout
        )

        # Compatibility with models that name dropout differently
        self.config.dropout = self.hparams.dropout

        encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

        self.model = DepressionDetectionModel(
            encoder=encoder,
            config=self.config,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        self.class_names = [l.name for l in DepressionLabel]

        # if needed, we'll cache the tokenizer
        self.tokenizer = None

    def tokenize_and_predict(self, sentences: Iterable[str],
                             ) -> List[DepressionPrediction]:
        """
        for inference, it's convenient to have the model be able
        to take in sentences as input and handle tokenization
        :param sentences: iterable of sentence strings to predict on
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                use_fast=True,
            )
        tokenized = self.tokenizer(
            sentences,
            max_length=512,
            padding="longest",  # pad to the longest sequence in the batch
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        with torch.no_grad():
            output = self.forward((input_ids, attention_mask))

        predictions = []
        for sentence, logits, probs in zip(sentences,
                                           output.logits.tolist(),
                                           output.probabilities.tolist(),
                                           ):

            tups = list(zip(DepressionLabel, probs))
            best_label, best_prob = max(tups, key=itemgetter(1))

            prediction = DepressionPrediction(
                sentence=sentence,
                predicted_label=best_label.name,
                probability=best_prob,
                label_to_probability={label.name: prob for label, prob in tups}
            )
            predictions.append(prediction)

        return predictions

    def forward(self, batch) -> DetectionOutput:
        # if label in batch, grab it. Otherwise, label is empty list
        input_ids, attention_mask, *label = batch
        logits = self.model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)

        return DetectionOutput(
            logits=logits,
            probabilities=probabilities,
        )

    def step(self, batch, batch_idx):
        """
        share code between all steps
        """
        input_ids, attention_mask, labels = batch
        output = self(batch)
        predictions = output.logits.argmax(-1)
        loss = F.cross_entropy(output.logits, labels)
        return {
            "loss": loss,
            "logits": output.logits,
            "predictions": predictions,
            "labels": labels,
        }

    def training_step(self, batch, batch_idx):
        step_output = self.step(batch, batch_idx)
        self.log("train/loss", step_output["loss"], sync_dist=True)
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = self.step(batch, batch_idx)
        self.log("val/loss", step_output["loss"], sync_dist=True)
        return step_output

    def test_step(self, batch, batch_idx):
        step_output = self.step(batch, batch_idx)
        self.log("test/loss", step_output["loss"], sync_dist=True)
        return step_output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate)

    def epoch_end(self, outputs, data_set):
        predictions = torch.cat([o['predictions'] for o in outputs], 0)
        labels = torch.cat([o['labels'] for o in outputs], 0)
        prf1 = self._calculate_prf1(labels, predictions)

        for key, value in prf1.items():
            if key.endswith('none'):
                continue
            self.log(f'{data_set}/{key}', value)

        figure = plot_prf1_per_class(prf1, class_names=self.class_names)
        self.logger.experiment.add_figure(f"{data_set}/"
                                          f"epoch={self.current_epoch}_"
                                          f"prf1-per-class",
                                          figure)

        loss = sum([o['loss'] for o in outputs])
        self.log(f'{data_set}/epoch_loss', loss)

        figure = plot_confusion_matrix(labels, predictions,
                                       class_names=self.class_names)
        self.logger.experiment.add_figure(
            f'{data_set}/epoch={self.current_epoch}_cm', figure)

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, 'train')
        if self.current_epoch == 0:
            self.model.make_computational_graph(logger=self.logger,
                                                device=self.device)

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, 'test')

    def _calculate_prf1(self, y_true, y_pred):
        results_pl = {}
        for average_type in ['weighted', 'micro', 'macro', 'none']:
            precision = torchmetrics.functional.classification.precision(
                y_pred,
                y_true,
                num_classes=len(self.class_names),
                average=average_type,
            )

            recall = torchmetrics.functional.classification.recall(
                y_pred,
                y_true,
                num_classes=len(self.class_names),
                average=average_type,
            )

            f1 = torchmetrics.functional.fbeta_score(
                y_pred,
                y_true,
                beta=.5,
                num_classes=len(self.class_names),
                average=average_type,
            )
            results_pl[f'p-{average_type}'] = precision
            results_pl[f'r-{average_type}'] = recall
            results_pl[f'f1-{average_type}'] = f1
        acc = torchmetrics.functional.classification.accuracy(y_pred, y_true)
        results_pl['acc'] = acc

        return results_pl
