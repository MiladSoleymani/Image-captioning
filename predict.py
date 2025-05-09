"""Batch caption generator. Usage: `python predict.py --ckpt model.ckpt img1.jpg img2.jpg`"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf

from img_cap.models.lightning_module import CaptionLightningModule
from img_cap.data.transforms import get_feature_extractor


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint")
    p.add_argument("images", nargs="+", help="Image paths")
    return p.parse_args()


def main():
    args = cli()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.create({"lr": 1e-5})  # dummy cfg – not used for inference
    model = (
        CaptionLightningModule.load_from_checkpoint(args.ckpt, cfg=cfg)
        .to(device)
        .eval()
    )
    feat_extractor = get_feature_extractor()

    for img in args.images:
        img_path = Path(img)
        image = Image.open(img_path).convert("RGB")
        pixel = feat_extractor(images=image, return_tensors="pt")["pixel_values"].to(
            device
        )
        caption = model.model.generate(pixel, tokenizer=model.tokenizer)
        print(f"{img_path.name}: {caption}")


if __name__ == "__main__":
    main()
