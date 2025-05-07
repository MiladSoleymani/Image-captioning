import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    SwinModel, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    ViTFeatureExtractor,
    get_linear_schedule_with_warmup
)
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import spacy

# Load spaCy model - lightweight English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model isn't installed, download it
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class ImageCaptioningDataset(Dataset):
    def __init__(self, data_file, feature_extractor, tokenizer, max_length=50):
        # Load data from the JSON file
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.images = data['images']
        self.annotations = data['annotations']
        
        # Create image_id to caption mapping
        self.image_id_to_caption = {}
        for ann in self.annotations:
            self.image_id_to_caption[ann['image_id']] = ann['caption']
        
        # Create list of valid images with captions
        self.valid_images = []
        for img in self.images:
            if img['id'] in self.image_id_to_caption:
                self.valid_images.append(img)
        
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img = self.valid_images[idx]
        img_url = img['coco_url']
        caption = self.image_id_to_caption[img['id']]
        
        # Download and process image
        try:
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_features = self.feature_extractor(images=image, return_tensors="pt")
            image_features = {k: v.squeeze(0) for k, v in image_features.items()}
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            # Return a placeholder for failed images
            image_features = {k: torch.zeros(v.shape[1:]) for k, v in 
                             self.feature_extractor(images=Image.new('RGB', (224, 224)), return_tensors="pt").items()}
        
        # Process caption
        caption_encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        caption_encoding = {k: v.squeeze(0) for k, v in caption_encoding.items()}
        
        return {
            "image_features": image_features,
            "input_ids": caption_encoding["input_ids"],
            "attention_mask": caption_encoding["attention_mask"]
        }

class ImageCaptioningModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder, feature_dim=768, hidden_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        
        # Freeze vision encoder parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # Projection layer to align vision features with text embedding dimensions
        self.projection = nn.Linear(vision_encoder.config.hidden_size, hidden_dim)
        
    def forward(self, pixel_values, input_ids, attention_mask):
        # Extract vision features
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Project vision features
        projected_features = self.projection(vision_features).unsqueeze(1)  # Add sequence dimension
        
        # Prepare text inputs with projected vision features as initial token
        extended_input_ids = torch.cat([torch.ones_like(input_ids[:, :1]) * self.text_decoder.config.bos_token_id, input_ids[:, :-1]], dim=1)
        
        # Pass to text decoder
        outputs = self.text_decoder(
            input_ids=extended_input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=projected_features
        )
        
        return outputs.logits
    
    def generate_caption(self, pixel_values, tokenizer, max_length=50):
        # Extract vision features
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            vision_features = vision_outputs.last_hidden_state[:, 0, :]
            projected_features = self.projection(vision_features).unsqueeze(1)
        
        # Generate text with vision features as context
        input_ids = torch.tensor([[tokenizer.bos_token_id]])
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.text_decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=projected_features
                )
            
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
        return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=5, learning_rate=5e-5):
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch["image_features"]["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                pixel_values = batch["image_features"]["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    all_bleu_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["image_features"]["pixel_values"].to(device)
            reference_captions = batch["input_ids"]
            
            # Generate captions
            generated_captions = []
            for i in range(len(pixel_values)):
                caption = model.generate_caption(
                    pixel_values=pixel_values[i:i+1], 
                    tokenizer=tokenizer
                )
                generated_captions.append(caption)
            
            # Calculate BLEU scores
            for i, gen_caption in enumerate(generated_captions):
                reference = tokenizer.decode(reference_captions[i].tolist(), skip_special_tokens=True)
                
                # Use spaCy tokenizer
                try:
                    reference_doc = nlp(reference.lower())
                    hypothesis_doc = nlp(gen_caption.lower())
                    
                    reference_tokens = [token.text for token in reference_doc]
                    hypothesis_tokens = [token.text for token in hypothesis_doc]
                    
                    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
                except Exception as e:
                    print(f"Error calculating BLEU score: {e}")
                    bleu_score = 0.0
                
                all_bleu_scores.append(bleu_score)
    
    avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0.0
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    return avg_bleu

def predict_caption(model, image_url, feature_extractor, tokenizer, device):
    # Download and process image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Extract features
    image_features = feature_extractor(images=image, return_tensors="pt")
    pixel_values = image_features["pixel_values"].to(device)
    
    # Generate caption
    caption = model.generate_caption(pixel_values=pixel_values, tokenizer=tokenizer)
    
    return caption, image

def main():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # انتخاب دستگاه مناسب
    print(f"Using device: {device}")


    
    # Initialize models and tokenizers
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    vision_encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text_decoder = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Special tokens
    special_tokens = {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
    tokenizer.add_special_tokens(special_tokens)
    text_decoder.resize_token_embeddings(len(tokenizer))
    
    # Create model
    model = ImageCaptioningModel(vision_encoder, text_decoder)
    model.to(device)
    
    # Create dataset and dataloaders
    data_file = "data.json"  # Update with your data file path
    
    full_dataset = ImageCaptioningDataset(data_file, feature_extractor, tokenizer)
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_dataloader, val_dataloader, device, num_epochs=5
    )
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_losses.png')
    
    # Evaluate model
    bleu_score = evaluate_model(model, val_dataloader, tokenizer, device)
    
    # Save model
    torch.save(model.state_dict(), "image_captioning_model.pth")
    print("Model saved!")
    
    # Example prediction
    if len(full_dataset.valid_images) > 0:
        sample_image_url = full_dataset.valid_images[0]['coco_url']
        caption, image = predict_caption(model, sample_image_url, feature_extractor, tokenizer, device)
        print(f"Generated Caption: {caption}")
        
        # Display image and caption
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(caption)
        plt.axis('off')
        plt.savefig('sample_prediction.png')

if __name__ == "__main__":
    main() 