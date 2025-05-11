from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTFeatureExtractor

from .coco_dataset import COCOCaptionDataset


class COCODataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_json: str,
        val_json: str,
        train_imgs: str,
        val_imgs: str,
        batch_size: int = 16,
        num_workers: int = 4,
        max_length: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
        )
        self.feat_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        self.ds_train: Optional[COCOCaptionDataset] = None
        self.ds_val: Optional[COCOCaptionDataset] = None

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.ds_train = COCOCaptionDataset(
                caption_file=self.hparams.train_json,
                image_dir=self.hparams.train_imgs,
                feature_extractor=self.feat_extractor,
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
            )
            self.ds_val = COCOCaptionDataset(
                caption_file=self.hparams.val_json,
                image_dir=self.hparams.val_imgs,
                feature_extractor=self.feat_extractor,
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
            )

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
