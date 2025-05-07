import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import (
    SwinModel,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    ViTFeatureExtractor,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import spacy
from nltk.translate.bleu_score import sentence_bleu
from image_captioning import (
    ImageCaptioningModel,
    train_model,
    evaluate_model,
    predict_caption,
)
from data.coco_dataset import get_coco_datasets

# Ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess

    print("Downloading spaCy English model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def main():
    # COCO dataset paths
    train_caption_file = (
        r"C:\Users\Keyvan\Desktop\New folder (11)\captions_train2017.json"
    )
    val_caption_file = r"C:\Users\Keyvan\Desktop\New folder (11)\captions_val2017.json"
    train_image_dir = r"C:\Users\Keyvan\Desktop\New folder (11)\train2017"
    val_image_dir = r"C:\Users\Keyvan\Desktop\New folder (11)\val2017"

    # Define device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # انتخاب دستگاه مناسب
    print(f"Using device: {device}")

    # Initialize models and tokenizers
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    vision_encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Initialize GPT-2 with cross-attention enabled
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_config.add_cross_attention = True  # Enable cross-attention
    text_decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=gpt2_config)

    # Special tokens
    special_tokens = {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
    tokenizer.add_special_tokens(special_tokens)
    text_decoder.resize_token_embeddings(len(tokenizer))

    # Create model
    model = ImageCaptioningModel(vision_encoder, text_decoder)
    model.to(device)

    # Create dataset and dataloaders
    print("Loading COCO datasets...")
    train_dataset, val_dataset = get_coco_datasets(
        train_caption_file,
        val_caption_file,
        train_image_dir,
        val_image_dir,
        feature_extractor,
        tokenizer,
        max_length=50,
    )

    # For faster testing, use only a subset of the data
    # Comment these lines for full training
    train_dataset_subset = torch.utils.data.Subset(
        train_dataset, indices=list(range(min(10000, len(train_dataset))))
    )
    val_dataset_subset = torch.utils.data.Subset(
        val_dataset, indices=list(range(min(1000, len(val_dataset))))
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset_subset,  # Use train_dataset for full training
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    val_dataloader = DataLoader(
        val_dataset_subset,  # Use val_dataset for full validation
        batch_size=16,
        num_workers=4,
    )

    print(
        f"Training with {len(train_dataloader.dataset)} samples, validating with {len(val_dataloader.dataset)} samples"
    )

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Train model
    train_losses, val_losses = train_model(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_epochs=3,
        learning_rate=2e-5,
    )

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("output/training_losses.png")

    # Evaluate model
    print("Evaluating model...")
    bleu_score = evaluate_model(model, val_dataloader, tokenizer, device)

    # Save model
    torch.save(model.state_dict(), "output/coco_image_captioning_model.pth")
    print(f"Model saved! Final BLEU Score: {bleu_score:.4f}")

    # Example predictions
    print("Generating sample predictions...")
    sample_results = []

    # Get 5 random samples from validation set
    val_indices = list(range(len(val_dataloader.dataset)))
    random_indices = val_indices[:5] if len(val_indices) >= 5 else val_indices

    for i in random_indices:
        sample = val_dataloader.dataset[i]
        img_id = sample["image_id"]
        true_caption = sample["caption"]

        # Generate caption
        pixel_values = sample["image_features"]["pixel_values"].unsqueeze(0).to(device)
        generated_caption = model.generate_caption(
            pixel_values=pixel_values, tokenizer=tokenizer
        )

        sample_results.append(
            {
                "image_id": img_id,
                "true_caption": true_caption,
                "generated_caption": generated_caption,
            }
        )

        print(f"Sample {i}:")
        print(f"  True caption: {true_caption}")
        print(f"  Generated: {generated_caption}")
        print()

    # Save sample results
    with open("output/sample_predictions.txt", "w") as f:
        for i, result in enumerate(sample_results):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  True caption: {result['true_caption']}\n")
            f.write(f"  Generated: {result['generated_caption']}\n\n")

    print("Finished! Check the 'output' directory for results.")


if __name__ == "__main__":
    main()
