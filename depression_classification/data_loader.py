import io
import logging
import os
import pathlib
from enum import (
    IntEnum,
    Enum,
)
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from cloudpathlib import CloudPath
from smart_open import smart_open
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)
from transformers import (
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


class DepressionLabel(IntEnum):
    """
    Enum to represent all possible types
    """
    NOT_DEPRESSION = 0
    DEPRESSION = 1


class DataFrameColumns(str, Enum):
    """
    enum to represent the column names of the csv storing the training,
    val, and test data. Only text and label are used, but id is present
    for any possible refactoring of the dataset
    """
    ID = "id"
    LABEL = "is_depression"
    TEXT = "clean_text"


class DepressionDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 cache_dir: str,
                 model_name_or_path: str,
                 max_seq_length: int = 128,
                 batch_size: int = 32,
                 ):
        super(DepressionDataModule, self).__init__()
        if data_dir.startswith("s3:/"):
            self.data_path = CloudPath(data_dir)
        else:
            self.data_path = pathlib.Path(data_dir)

        if cache_dir.startswith("s3:/"):
            self.cache_path = CloudPath(cache_dir)
        else:
            self.cache_path = pathlib.Path(cache_dir)
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                       use_fast=True)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _df_to_tensor_dataset(self, df: pd.DataFrame) -> TensorDataset:
        texts = df[DataFrameColumns.TEXT].astype(str).tolist()
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        try:
            # todo - use the enum
            # seq_cls_labels = [DepressionLabel[lab].value
            #                   for lab in df[DataFrameColumns.LABEL]]
            seq_cls_labels = [lab for lab in df[DataFrameColumns.LABEL]]
        except ValueError as e:
            raise ValueError("Encountered invalid label value") from e

        seq_cls_labels_tens = torch.tensor(
            [lab for lab in seq_cls_labels],
            dtype=torch.long,
        )

        dataset = TensorDataset(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            seq_cls_labels_tens,
        )

        return dataset

    def _load_from_cache(self, split: str):
        cache_path = self.cache_path / f"{split}.pt"

        if not os.path.exists(cache_path):
            logger.info(f"Loading & tokenizing {split} set")
            read_from_loc = self.data_path / f"{split}.csv"
            df = pd.read_csv(read_from_loc)
            ret = self._df_to_tensor_dataset(df)
            torch.save(ret, str(cache_path))
            return ret

        logger.info(f"Loading cached {split} set from {cache_path}")
        # if reading from s3, need this
        with smart_open(str(cache_path), 'rb') as f:
            buffer = io.BytesIO(f.read())
        tensor_dataset = torch.load(buffer)
        return tensor_dataset

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit", "validate") and \
                self.train_dataset is None:
            self.train_dataset = self._load_from_cache("train")
            self.val_dataset = self._load_from_cache("val")

        if stage == "test" and self.test_dataset is None:
            self._load_from_cache("test")

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
        )
        return dataloader
