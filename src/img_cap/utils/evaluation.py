from __future__ import annotations

from typing import List

from nltk.translate.bleu_score import sentence_bleu
import spacy

_NLP = spacy.blank("en")


def _tok(text: str):
    return [t.text for t in _NLP(text.lower())]


def bleu_score_batch(model, batch, device) -> List[float]:
    pix = batch["pixel_values"].to(device)
    refs = batch["input_ids"]
    tokenizer = model.tokenizer
    scores: List[float] = []
    for i in range(pix.size(0)):
        ref = tokenizer.decode(refs[i], skip_special_tokens=True)
        hyp = model.generate(pix[i : i + 1], tokenizer)
        scores.append(sentence_bleu([_tok(ref)], _tok(hyp)))
    return scores
