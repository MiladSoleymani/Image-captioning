from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    SwinModel,
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
)


class ImageCaptioningModel(nn.Module):
    """Frozen Swin encoder + GPT‑2 decoder with cross‑attention."""

    def __init__(self, vision_encoder, text_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder.eval()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        self.text_decoder = text_decoder
        self.proj = nn.Linear(
            vision_encoder.config.hidden_size, text_decoder.config.hidden_size
        )

    # ------------------------------------------------------------------
    def forward(self, pixel_values, input_ids, attention_mask):
        vis_out = self.vision_encoder(pixel_values=pixel_values)
        feats = self.proj(vis_out.last_hidden_state[:, 0, :]).unsqueeze(1)
        logits = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=feats,
        ).logits
        return logits

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, pixel_values, tokenizer, max_length: int = 50):
        device = pixel_values.device
        feats = self.proj(
            self.vision_encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        ).unsqueeze(1)
        ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        for _ in range(max_length):
            logits = self.text_decoder(
                input_ids=ids, encoder_hidden_states=feats
            ).logits
            next_tok = logits[:, -1, :].argmax(-1)
            ids = torch.cat([ids, next_tok.unsqueeze(-1)], -1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    @staticmethod
    def build_pretrained():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
        )

        vision = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        cfg = GPT2Config.from_pretrained("gpt2")
        cfg.add_cross_attention = True
        text = GPT2LMHeadModel.from_pretrained("gpt2", config=cfg)
        text.resize_token_embeddings(len(tokenizer))
        model = ImageCaptioningModel(vision, text)
        model.tokenizer = tokenizer  # handy reference
        return model
