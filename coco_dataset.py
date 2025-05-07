import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class COCOCaptioningDataset(Dataset):
    def __init__(
        self, caption_file, image_dir, feature_extractor, tokenizer, max_length=50
    ):
        # Load data from the COCO JSON file
        with open(caption_file, "r") as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = data["annotations"]
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Filter out images without file_name or captions
        valid_annotations = []
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id in self.images and "file_name" in self.images[img_id]:
                valid_annotations.append(ann)

        self.annotations = valid_annotations
        print(f"Loaded {len(self.annotations)} valid annotations from {caption_file}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_id = ann["image_id"]
        caption = ann["caption"]

        # Get image path
        img_filename = self.images[img_id]["file_name"]
        img_path = os.path.join(self.image_dir, img_filename)

        # Load and preprocess image
        try:
            image = Image.open(img_path).convert("RGB")
            image_features = self.feature_extractor(images=image, return_tensors="pt")
            image_features = {k: v.squeeze(0) for k, v in image_features.items()}
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Return a placeholder for failed images
            image_features = {
                k: torch.zeros(v.shape[1:])
                for k, v in self.feature_extractor(
                    images=Image.new("RGB", (224, 224)), return_tensors="pt"
                ).items()
            }

        # Process caption
        caption_encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        caption_encoding = {k: v.squeeze(0) for k, v in caption_encoding.items()}

        return {
            "image_features": image_features,
            "input_ids": caption_encoding["input_ids"],
            "attention_mask": caption_encoding["attention_mask"],
            "caption": caption,
            "image_id": img_id,
        }


def get_coco_datasets(
    train_caption_file,
    val_caption_file,
    train_image_dir,
    val_image_dir,
    feature_extractor,
    tokenizer,
    max_length=50,
):
    """
    Create train and validation datasets for COCO
    """
    train_dataset = COCOCaptioningDataset(
        train_caption_file, train_image_dir, feature_extractor, tokenizer, max_length
    )

    val_dataset = COCOCaptioningDataset(
        val_caption_file, val_image_dir, feature_extractor, tokenizer, max_length
    )

    return train_dataset, val_dataset
