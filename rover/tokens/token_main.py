import fire
import transformers as tr
from loguru import logger

from rover.data_utils.io_helpers import human_readable
from rover.data_utils.loader import load_data, save_data


def tokenize(
    tokenizer_name_path: str,
    input_path: str,
    output_path: str,
    *,
    column_name="text",
    max_records: int = None,
    counts_column_name="token_count",
    output_column_name="input_ids",
    input_format: str = "parquet",
    output_format: str = "parquet",
    add_special_tokens=False,
    overwrite=False,
    batch_size=1024,
):

    tok = tr.AutoTokenizer.from_pretrained(tokenizer_name_path)
    data = load_data(input_path, input_format)

    if not output_column_name:
        output_column_name = f"{column_name}_input_ids"

    if output_column_name in data.column_names:
        raise ValueError(f"dataset already contains column {output_column_name}")

    if counts_column_name in data.column_names:
        raise ValueError(f"dataset already contains column {counts_column_name}")

    def f(xs):
        tokens = tok(xs, add_special_tokens=add_special_tokens)["input_ids"]
        out = {output_column_name: tokens}
        if counts_column_name:
            out[counts_column_name] = [len(t) for t in tokens]
        return out

    if max_records:
        logger.warning(f"restricting dataset to {max_records} records")
        sample_count = min(max_records, len(data))
        data = data.select(range(sample_count))

    data = data.map(
        f,
        batched=True,
        batch_size=batch_size,
        input_columns=[column_name],
    )

    save_data(
        data, output_path=output_path, output_format=output_format, overwrite=overwrite
    )

    sample_count = min(10, len(data))
    sample_data = data.select(range(sample_count)).to_pandas()
    logger.info(f"{sample_data}")

    logger.info(f"rows = {human_readable(len(data))}")

    if counts_column_name:
        data.set_format("np", columns=[counts_column_name])
        counts = data.select_columns([counts_column_name]).to_pandas()
        token_count = counts[counts_column_name].sum()
        logger.info(f"tokens = {human_readable(token_count)}")
        return token_count
    return -1


if __name__ == "__main__":
    fire.Fire(dict(tokenize=tokenize))
