import random
import typing as ty
import warnings

import datasets as ds
import fire
import torch
import trl
from loguru import logger
from torch.utils.data import IterableDataset

from rover.data_utils.loader import load_data
from rover.workflows.pretrain.model_factory import load_tokenizer


class DummyTokenizer:
    def __init__(self, bos_token_id):
        self.eos_token_id = 0
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids_list, add_special_tokens, truncation):
        return dict(input_ids=[[self.bos_token_id] + xs for xs in input_ids_list])


def packed_dataset(
    data: ty.Union[ds.Dataset, ds.DatasetDict],
    eos_token_id: int,
    bos_token_id: int,
    seq_length=2048,
):
    if isinstance(data, ds.DatasetDict):
        return {
            k: packed_dataset(v, eos_token_id, bos_token_id, seq_length)
            for k, v in data.items()
        }

    return MyConstantLengthDataset(
        dataset=data,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        seq_length=seq_length,
        num_of_sequences=1024,
        shuffle=False,
    )


class MyConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

    Args:
        tokenizer (`transformers.PreTrainedTokenizer`):
            The processor used for processing the data_utils.
        dataset (`dataset.Dataset`):
            Dataset with text files.
        dataset_text_field (`Optional[str]`, *optional*, defaults to `None`):
            Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
        formatting_func (`Callable`, *optional*):
            Function that formats the text before tokenization. Usually it is recommended to have follows a certain
            pattern such as `"### Question: {question} ### Answer: {answer}"`
        infinite (`bool`, *optional*, defaults to `False`):
            If True the iterator is reset after dataset reaches end else stops.
        seq_length (`int`, *optional*, defaults to `1024`):
            Length of token sequences to return.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of token sequences to keep in buffer.
        chars_per_token (`int`, *optional*, defaults to `3.6`):
            Number of characters per token used to estimate number of tokens in text buffer.
        bos_token_id (`int`, *optional*, defaults to `0`):
            Id of the end of sequence token if the passed tokenizer does not have an EOS token.
        shuffle (`bool`, *optional*, defaults to `True`)
            Shuffle the examples before they are returned
        append_concat_token (`bool`, *optional*, defaults to `True`)
            If true, appends `eos_token_id` at the end of each sample being packed.
        add_special_tokens (`bool`, *optional*, defaults to `True`)
            If true, tokenizers adds special tokens to each sample being packed.
    """

    def __init__(
        self,
        dataset,
        bos_token_id,
        eos_token_id,
        infinite=False,
        seq_length=2048,
        num_of_sequences=1024,
        shuffle=True,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * num_of_sequences
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append((next(iterator))["input_ids"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn(
                            "The dataset reached end and the iterator is reset to the start."
                        )
                    else:
                        more_examples = False
                        break
            if self.shuffle:
                random.shuffle(buffer)

            tokenized_inputs = buffer
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                tokenized_input = tokenized_input + [
                    self.eos_token_id,
                    self.bos_token_id,
                ]
                all_token_ids.extend(tokenized_input)
            examples = []
            i = 0
            while i < len(all_token_ids):
                head = all_token_ids[i]
                if head == self.bos_token_id:
                    xs = all_token_ids[i : i + self.seq_length]
                    i += self.seq_length
                else:
                    xs = [self.bos_token_id] + all_token_ids[
                        i : i + self.seq_length - 1
                    ]
                    i += self.seq_length - 1

                if len(xs) == self.seq_length:
                    examples.append(xs)
                else:
                    # discard the leftovers in the batch. ensure max_buffer_size
                    # is large enough so this is not an issue
                    pass

            if self.shuffle:
                # Shuffle again, otherwise split examples occur in consecutive tensors.
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def make_constant(
    data: ty.Union[ds.Dataset, ds.DatasetDict, str],
    *,
    output_dir: str = None,
    seq_length=2048,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    input_format: str = "",
    max_records: int = None,
    **kwargs
) -> ty.Union[ds.Dataset, ds.DatasetDict]:

    if isinstance(data, str):
        data = load_data(input_path=data, input_format=input_format, **kwargs)

    tokenizer = load_tokenizer(tokenizer_name)

    def make_constant_single(single_data):
        if max_records and len(single_data) > max_records:
            single_data = single_data.select(range(max_records))

        return ds.Dataset.from_list(
            list(
                trl.trainer.ConstantLengthDataset(
                    tokenizer=tokenizer,
                    dataset=single_data,
                    seq_length=seq_length,
                    dataset_text_field="text",
                    num_of_sequences=1024 * 64,
                    append_concat_token=False,
                )
            )
        )

    if isinstance(data, ds.DatasetDict):
        output = ds.DatasetDict({s: make_constant_single(data[s]) for s in data.keys()})
    elif isinstance(data, ds.Dataset):
        output = make_constant_single(data)
    else:
        raise ValueError("")

    logger.info(output_dir)
    if output_dir:
        output.save_to_disk(output_dir)
    else:
        return output


if __name__ == "__main__":
    fire.Fire(make_constant)
