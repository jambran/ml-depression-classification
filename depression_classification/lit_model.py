import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from .data_loader import DepressionLabel
from .data_models import (
    DetectionOutput,
)
from .depression_detection_config import DepressionDetectionConfig
from .model import DepressionDetectionModel
from .plotting_metrics import (
    plot_prf1_per_class,
    plot_confusion_matrix,
)


class LitDepressionDetectionModel(pl.LightningModule):

    def __init__(self,
                 experiment_name: str,
                 model_name_or_path: str,
                 learning_rate: float,
                 dropout,
                 attention_dropout,
                 num_layers,
                 hidden_size,
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.class_names = [l.name for l in DepressionLabel]

        self.config = DepressionDetectionConfig(
            model_name_or_path,
            num_layers=self.hparams.num_layers,
            num_classes=len(self.class_names),
            dropout=self.hparams.dropout,
            attention_dropout=self.hparams.attention_dropout,
        )

        self.model = DepressionDetectionModel(self.config)

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
