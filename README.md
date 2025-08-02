# Image Captioning

A PyTorch Lightning implementation of an image captioning model using Swin Transformer as the vision encoder and GPT-2 as the text decoder.

## Overview

This project implements an image captioning pipeline that generates natural language descriptions for images. The model architecture combines:
- **Vision Encoder**: Frozen Swin-Tiny transformer for extracting visual features
- **Text Decoder**: GPT-2 with cross-attention for generating captions
- **Training Framework**: PyTorch Lightning with Hydra configuration management

## Requirements

- Python >= 3.10
- PyTorch
- PyTorch Lightning
- Transformers
- Hydra
- Additional dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Image-captioning

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Project Structure

```
Image-captioning/
├── conf/                    # Hydra configuration files
│   ├── config.yaml         # Main config
│   ├── datamodule/         # Data module configs
│   ├── model/              # Model configs
│   ├── optimizer/          # Optimizer configs
│   └── trainer/            # Trainer configs
├── src/img_cap/            # Source code
│   ├── data/               # Dataset and datamodule
│   ├── models/             # Model architecture
│   └── utils/              # Utilities (evaluation, transforms)
├── train.py                # Training script
├── predict.py              # Inference script
└── tests/                  # Unit tests
```

## Training

Train the model using the provided configuration:

```bash
python train.py
```

### Configuration

The project uses Hydra for configuration management. You can override any configuration parameter:

```bash
# Change batch size
python train.py datamodule.batch_size=64

# Use different learning rate
python train.py model.lr=1e-4

# Modify trainer settings
python train.py trainer.max_epochs=50
```

## Inference

Generate captions for your images:

```bash
python predict.py --ckpt path/to/checkpoint.ckpt image1.jpg image2.jpg
```

## Model Architecture

The model uses a vision-language architecture:

1. **Vision Encoder**: Swin-Tiny transformer (frozen during training)
   - Pre-trained on ImageNet
   - Extracts visual features from input images

2. **Projection Layer**: Linear layer to map visual features to text embedding space

3. **Text Decoder**: GPT-2 with cross-attention
   - Generates captions conditioned on visual features
   - Uses special tokens: [BOS], [EOS], [PAD]

## Dataset

The project is configured to work with the COCO dataset by default. The data module handles:
- Image preprocessing and augmentation
- Caption tokenization
- Batch collation with proper padding

## Evaluation

The model is evaluated using BLEU score during validation. The best checkpoints are saved based on validation BLEU performance.

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information if applicable]
```