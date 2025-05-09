from __future__ import annotations

import torch
import pytorch_lightning as pl
from torch.optim import AdamW

from img_cap.utils.evaluation import bleu_score_batch
from .caption_model import ImageCaptioningModel


class CaptionLightningModule(pl.LightningModule):
    def __init__(self, lr: float = 5e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = ImageCaptioningModel.build_pretrained()
        self.tokenizer = self.model.tokenizer
        self.criterion = torch.nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def forward(self, pixel_values, input_ids, attention_mask):
        return self.model(pixel_values, input_ids, attention_mask)

    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        logits = self(**batch)
        loss = self.criterion(
            logits.view(-1, logits.size(-1)), batch["input_ids"].view(-1)
        )
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        logits = self(**batch)
        loss = self.criterion(
            logits.view(-1, logits.size(-1)), batch["input_ids"].view(-1)
        )
        bleu = torch.tensor(bleu_score_batch(self.model, batch, self.device)).mean()
        self.log_dict({"val/loss": loss, "val/bleu": bleu}, prog_bar=True)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)
