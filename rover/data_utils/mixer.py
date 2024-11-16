from dataclasses import dataclass

import datasets as ds
from loguru import logger
import transformers as tr

from rover.data_utils.loader import load_data
from rover.tokens.constant_length import packed_dataset


@dataclass
class DataResult:
    data: ds.DatasetDict
    collator: tr.DataCollator


def simple_dataset(
    data: ds.DatasetDict,
    tokenizer,
    rank: int,
    make_constant_length: int = None,
) -> DataResult:

    # todo:
    # if (not make_constant_length) and (rank == 0):
    #     seq_length = next(iter(data["train"]))["input_ids"]
    #     token_counts = {k: seq_length * len(data[k]) for k in data.keys()}
    #     logger.info(f"sequence length {seq_length}, token counts {token_counts}")

    collator = None
    if make_constant_length is not None:
        # todo: this should ideally be a collator instead of new dataset
        data = {
            k: packed_dataset(
                data[k],
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                seq_length=make_constant_length,
            )
            for k in data.keys()
        }
    else:
        # packed dataset internally converts to torch tensors
        # for vanilla case we convert to numpy
        [d.set_format("np") for d in data.values()]

    if rank == 0:
        logger.info(f"data info dataset={data}")

    return DataResult(data, collator=collator)
