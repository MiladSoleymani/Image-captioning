import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import spacy
from transformers import (
    SwinModel, 
    AutoTokenizer, 
    GPT2Config,
    GPT2LMHeadModel, 
    ViTFeatureExtractor
)
from image_captioning import ImageCaptioningModel

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    print("Downloading spaCy English model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def load_model(model_path):
    """
    Load the trained image captioning model
    """
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models and tokenizers
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
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
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, feature_extractor, tokenizer, device

def analyze_caption(caption):
    """
    Analyze a caption using spaCy to provide linguistic insights
    """
    doc = nlp(caption)
    
    # Extract entities if any
    entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
    
    # Extract noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    # Part of speech info
    pos_info = [(token.text, token.pos_) for token in doc]
    
    return {
        "entities": entities,
        "noun_phrases": noun_phrases,
        "pos_info": pos_info
    }

def generate_caption_for_image(image_path, model, feature_extractor, tokenizer, device):
    """
    Generate a caption for a single image
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_features = feature_extractor(images=image, return_tensors="pt")
    pixel_values = image_features["pixel_values"].to(device)
    
    # Generate caption
    with torch.no_grad():
        caption = model.generate_caption(pixel_values=pixel_values, tokenizer=tokenizer)
    
    # Analyze caption with spaCy
    analysis = analyze_caption(caption)
    
    return caption, image, analysis

def main():
    # Path to the trained model
    model_path = "output/coco_image_captioning_model.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first using train_coco.py")
        return
    
    # Create output directory
    os.makedirs("predictions", exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, feature_extractor, tokenizer, device = load_model(model_path)
    
    # Get input image
    while True:
        image_path = input("Enter path to an image (or 'q' to quit): ")
        
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"Image not found at {image_path}. Please enter a valid path.")
            continue
        
        try:
            # Generate caption
            caption, image, analysis = generate_caption_for_image(
                image_path, model, feature_extractor, tokenizer, device
            )
            
            print(f"Generated Caption: {caption}")
            print("\nCaption Analysis:")
            print(f"  Noun phrases: {', '.join(analysis['noun_phrases']) if analysis['noun_phrases'] else 'None'}")
            print(f"  Entities: {', '.join(analysis['entities']) if analysis['entities'] else 'None'}")
            
            # Display image and caption
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.title(caption)
            plt.axis('off')
            
            # Save prediction
            filename = os.path.basename(image_path)
            output_path = os.path.join("predictions", f"prediction_{filename}")
            plt.savefig(output_path)
            print(f"Prediction saved to {output_path}")
            
            # Show image
            plt.show()
            
        except Exception as e:
            print(f"Error processing image: {e}")

if __name__ == "__main__":
    main() 