import typing as ty

import torch
import transformers as tr
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

#
#
# def pad_with_labels(
#     msg: ty.Dict[str, ty.List], tokenizer: tr.PreTrainedTokenizer, max_length: int
# ):
#     # todo: filter out max_length msgs
#
#     out = pad_without_fast_tokenizer_warning(
#         tokenizer,
#         encoded_inputs=dict(input_ids=msg["input_ids"]),
#         padding="longest",
#         # pad_to_multiple_of=64,
#         return_attention_mask=True,
#         return_tensors="pt",
#     )
#
#     length = out["input_ids"].shape[-1]
#     labels = [xs + ([-100] * (length - len(xs))) for xs in msg["label"]]
#     out["label"] = torch.tensor(labels)
#     assert out["label"].shape == out["input_ids"].shape
#     return out
#
#


class PadWithLabelsCollator:
    def __init__(
        self, tokenizer: tr.PreTrainedTokenizer, max_length: int, pad_to_multiple_of=64
    ):
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        # todo: filter out max_length msgs

        out = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            encoded_inputs=dict(input_ids=[rec["input_ids"] for rec in batch]),
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        length = out["input_ids"].shape[-1]
        labels = [xs["label"] + ([-100] * (length - len(xs["label"]))) for xs in batch]
        out["labels"] = torch.tensor(labels)
        assert out["labels"].shape == out["input_ids"].shape
        return out
