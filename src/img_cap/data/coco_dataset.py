from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, PreTrainedTokenizer


class COCOCaptionDataset(Dataset):
    """
    Minimal COCO-style caption dataset that
    * keeps only captions whose image file actually exists
    * returns ViT-ready pixel tensors + tokenized caption
    """

    def __init__(
        self,
        caption_file: str | Path,
        image_dir: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 50,
        sample_one_caption: bool = True,
        processor_name: str = "google/vit-base-patch16-224",
    ) -> None:
        """
        Args
        ----
        caption_file : path to a COCO-format JSON (e.g. captions_train2017.json)
        image_dir    : directory containing the image files
        tokenizer    : PreTrainedTokenizer for captions
        max_length   : sentence length for tokenizer padding/truncation
        sample_one_caption : if True keep exactly one caption per image,
                             else keep all captions (flattened)
        processor_name : name or path for ViT-compatible AutoImageProcessor
        """
        super().__init__()

        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_processor = AutoImageProcessor.from_pretrained(processor_name)

        # ------------------------------------------------------------------
        # 1) Parse COCO JSON once
        with open(caption_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 2) Keep only images that truly exist on disk
        self.id2file: Dict[int, Path] = {}
        for img in meta["images"]:
            p = self.image_dir / img["file_name"]
            print(p)
            if p.is_file():  # ✅ file exists
                self.id2file[img["id"]] = p
            # else: silently drop or log; depends on your needs

        # 3) Build (image_id, caption) records
        if sample_one_caption:
            # ‣ one caption per annotation (same as original behaviour)
            self.records: List[Tuple[int, str]] = [
                (ann["image_id"], ann["caption"])
                for ann in meta["annotations"]
                if ann["image_id"] in self.id2file
            ]
        else:
            # ‣ all captions – still ensure image file exists
            self.records = [
                (ann["image_id"], ann["caption"])
                for ann in meta["annotations"]
                if ann["image_id"] in self.id2file
            ]

        if not self.records:
            raise RuntimeError(
                "No matching <image, caption> pairs found: "
                "check `caption_file`, `image_dir`, or missing files."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id, caption = self.records[idx]

        # ❶ Load & preprocess image
        image = Image.open(self.id2file[img_id]).convert("RGB")
        pix = self.img_processor(images=image, return_tensors="pt")

        # ❷ Tokenize caption
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # ❸ Package tensors (batch dim is squeezed to match original style)
        return {
            "pixel_values": pix["pixel_values"].squeeze(0),  # (3, 224, 224)
            "input_ids": text["input_ids"].squeeze(0),  # (L,)
            "attention_mask": text["attention_mask"].squeeze(0),
        }


def coco_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stacks each key in a batch of dicts.
    Use with DataLoader:  DataLoader(dataset, batch_size=…, collate_fn=coco_collate)
    """
    import torch

    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}
