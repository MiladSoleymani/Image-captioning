from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ViTFeatureExtractor


class COCOCaptionDataset(Dataset):
    """Minimal COCO caption dataset."""

    def __init__(
        self,
        caption_file: str | Path,
        image_dir: str | Path,
        feature_extractor: ViTFeatureExtractor,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 50,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

        meta = json.loads(Path(caption_file).read_text())
        images = {img["id"]: img for img in meta["images"]}
        self.records: List[Tuple[int, str]] = [
            (ann["image_id"], ann["caption"])
            for ann in meta["annotations"]
            if ann["image_id"] in images
        ]
        self.id2file: Dict[int, str] = {
            img_id: img["file_name"] for img_id, img in images.items()
        }

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_id, caption = self.records[idx]
        image = Image.open(self.image_dir / self.id2file[img_id]).convert("RGB")
        pix = self.feature_extractor(images=image, return_tensors="pt")
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        example: Dict[str, Any] = {
            "pixel_values": pix["pixel_values"].squeeze(0),
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
        }
        return example
