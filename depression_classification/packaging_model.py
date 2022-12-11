import logging
import os
import shutil

import pytorch_lightning as pl
import torch

"""
At the end of a model training job in sagemaker, everything located in 
/opt/ml/model
will be packaged into a `model.tar.gz` than can be used for deployment.

It's critical that everything needed for deployment is present in that
`model.tar.gz`, or else we will have errors arise when trying to deploy.

This file holds functions that move any other relevant code into 
/opt/ml/model so that the resulting `model.tar.gz` will have all that it 
needs.
"""

logger = logging.getLogger(__name__)


def list_files(startpath: str):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = "-" * 4 * (level)
        logger.info("{}{}/".format(indent, os.path.basename(root)))
        subindent = "-" * 4 * (level + 1)
        for f in files:
            logger.info("{}{}".format(subindent, f))


def save_model(trainer: pl.Trainer,
               model: pl.LightningModule,
               model_dir: str,
               ):
    """
    to save the model as a .tar.gz on s3, we must first save the model
    locally on the instance we're training on (usually /opt/ml/output/model)
    With this properly set, the estimator will wrap up the model's
    checkpoint file into a .tar.gz for deployment
    :param trainer: the pytorch lightning trainer
    :param model: the model to be saved
    :param model_dir: the location to save it (if running sagemaker, should be
                      /opt/ml/output/model)
    :return: None
    """
    # for debugging, list the file directory
    list_files("..")

    # keep hardcoded things in one place - easier to change later if need be
    model_pth = "model.pt"
    last_ckpt = "last.ckpt"
    code = "code"

    logging.info(f"Saving the model to {model_dir}")
    model_path = os.path.join(model_dir, model_pth)
    torch.save(model.model.to("cpu"), model_path)

    checkpoint_path = os.path.join(model_dir, last_ckpt)
    trainer.save_checkpoint(checkpoint_path)

    # copy all of code directory to opt/ml/model for easy deploy later
    opt_ml = f"opt/ml"
    if os.path.exists(opt_ml):
        shutil.copytree(f"{opt_ml}/{code}", f"{model_dir}/{code}")

    list_files("..")
