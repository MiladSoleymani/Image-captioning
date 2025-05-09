from transformers import ViTFeatureExtractor


def get_feature_extractor():
    return ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
