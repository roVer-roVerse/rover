from collections import defaultdict

import fire
from loguru import logger
import transformers as tr

import datasets as ds
from rover.data_utils.loader import load_data
from rover.data_utils.mixer import simple_dataset
from rover.workflows.pretrain.model_factory import (
    load_model,
    ModelArguments,
    load_tokenizer,
)
from rover.workflows.pretrain.training_loop import train


def load_datasets() -> ds.DatasetDict:
    data = defaultdict(list)
    d1 = load_data(
        input_path="./out/datasets/minipile/2k",
        input_format="arrow",
        remove_wrapper=False,
        streaming=True,
    )
    # d1 = d1.with_format("np", ["input_ids"])
    data["validation"].append(d1["validation"])
    data["test"].append(d1["test"])

    d2 = load_data(
        input_path="./out/datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2/token-llama3",
        input_format="parquet",
        remove_wrapper=False,
        streaming=True,
    )

    out = ds.DatasetDict()

    out["train"] = ds.interleave_datasets(
        datasets=[d1["train"].to_iterable_dataset(num_shards=32), d2["train"]],
        probabilities=[0.8, 0.2],
        seed=42,
    )

    if "validation" in data:
        out["validation"] = ds.concatenate_datasets(data["validation"])

    if "test" in data:
        out["test"] = ds.concatenate_datasets(data["test"])

    return out


def main(
    model_name_or_path_or_config: str,
    *,
    tokenizer_name: str = None,
    device="cuda",
    transformer_engine: bool = None,
    attn_implementation="flash_attention_2",
    seq_length: int = 2048,
    **kwargs,
):
    training_args = tr.Seq2SeqTrainingArguments(**kwargs)
    training_args.report_to = []
    logger.info(f"training args", training_args=training_args)

    if attn_implementation == "flash_attention_2":
        assert device == "cuda", f"FA2 is only supported on cuda"

    model_args = ModelArguments(
        model_name_or_path_or_config=model_name_or_path_or_config,
        tokenizer_name=tokenizer_name,
        te=transformer_engine,
        attn_implementation=attn_implementation,
    )

    model = load_model(model_args)
    tok = load_tokenizer(model_args.tokenizer_name)

    data = load_datasets()

    data = simple_dataset(
        data=data,
        tokenizer=tok,
        rank=training_args.local_rank,
        make_constant_length=seq_length,
    )

    train(
        data=data,
        model=model,
        tokenizer=tok,
        device=device,
        training_args=training_args,
    )


if __name__ == "__main__":
    fire.Fire(main)
