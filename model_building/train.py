from depression_classification.data_loader import (
    DepressionDataModule,
    DepressionLabel,
)
from depression_classification.packaging_model import save_model
import pytorch_lightning as pl
import torch
import argparse
import logging
from pytorch_lightning.loggers import TensorBoardLogger

from depression_classification.lit_model import LitDepressionDetectionModel

parser = argparse.ArgumentParser(description="Train a depression classifier")
parser.add_argument("--exp_name", type=str, default="full-dataset",
                    help="experiment saved under this name. "
                         "When name is 'debug', only one batch of data is "
                         "used.")
parser.add_argument("--model_name", type=str, default="distilbert-base-cased",
                    help="which encoder to use. See huggingface for options")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--dropout", type=float, default=.15)
parser.add_argument("--attention_dropout", type=float, default=.15)
parser.add_argument("--num_layers", type=int, default=1,
                    help="number of fully connected layers after the encoder "
                         "and before the final classification layer")
parser.add_argument("--epochs", type=int, default=5)

# Data, model, and output directories
parser.add_argument("--data_dir", type=str, default="../data",
                    help="path to directory containing train.csv, val.csv, "
                         "test.csv")
parser.add_argument("--cache_dir", type=str, default="../data/cache",
                    help="path to cached train, val, and test TensorDatasets")
parser.add_argument("--output_dir", type=str, default="./experiments/5-epochs")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--model_dir", type=str, default=None,
                    help="Defaults to trainer's root dir")
parser.add_argument("--load_from_checkpoint", type=str, default=None,
                    help="path to checkpoint that should be loaded before "
                         "model training begins")

if __name__ == "__main__":

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )

    data_module = DepressionDataModule(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        model_name_or_path=args.model_name,
        batch_size=args.batch_size,
    )

    model = LitDepressionDetectionModel(
        experiment_name=args.exp_name,
        model_name_or_path=args.model_name,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        num_layers=args.num_layers,
        hidden_size=128,
        num_classes=len(DepressionLabel),
    )
    if args.load_from_checkpoint:
        model = LitDepressionDetectionModel.load_from_checkpoint(
            args.load_from_checkpoint
        )

    if args.checkpoint_dir == "checkpoints":
        checkpoint_path = f"{args.output_dir}/{args.checkpoint_dir}"
    else:
        checkpoint_path = args.checkpoint_dir

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=f"checkpoint-depression-detection-"
                 f"{model.hparams.experiment_name}"
                 "-{epoch:02d}-{step}",
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        min_epochs=args.epochs,
        fast_dev_run=args.exp_name == "debug",
        logger=[TensorBoardLogger(args.output_dir,
                                  name="tensorboard",
                                  version=args.exp_name,
                                  ),
                ],
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=data_module)
    save_model(trainer, model,
               model_dir=args.model_dir if args.model_dir else trainer.log_dir)
