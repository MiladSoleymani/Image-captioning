"""Generic Hydra + Lightning training launcher."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):  # noqa: D401
    print(OmegaConf.to_yaml(cfg))

    # ---- instantiate objects from config
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)

    ckpt_cb = ModelCheckpoint(
        monitor="val/bleu",
        mode="max",
        filename="epoch{epoch:02d}-bleu{val/bleu:.3f}",
        save_top_k=3,
    )

    trainer: Trainer = instantiate(
        cfg.trainer, callbacks=[ckpt_cb, LearningRateMonitor()]
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
