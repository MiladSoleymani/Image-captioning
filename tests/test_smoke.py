"""Quick smoke test – import & forward pass"""

from pathlib import Path

import torch

from img_cap.models.caption_model import ImageCaptioningModel


def test_forward():
    m = ImageCaptioningModel.build_pretrained()
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_ids = torch.tensor([[m.tokenizer.bos_token_id] * 10])
    dummy_mask = torch.ones_like(dummy_ids)
    logits = m(dummy_img, dummy_ids, dummy_mask)
    assert logits.shape[0] == 1
    assert logits.shape[-1] == m.tokenizer.vocab_size
