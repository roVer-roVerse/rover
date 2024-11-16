import os
import re

import datasets as ds
import fire
from huggingface_hub import HfFileSystem
from loguru import logger
from tqdm import tqdm
import typing as ty

from transformers import TrainingArguments

from rover.data_utils.io_helpers import write_file, read_file


def hfls(
    dataset_name: str,
    *,
    prefix: str = None,
    output_file=None,
    input_format="parquet",
    recursive=False,
    **kwargs,
):
    fs = HfFileSystem()

    path = f"datasets/{dataset_name}"
    if prefix:
        path = f"{path}/{prefix}"
    if recursive:
        path += "/**"

    path += f"/*.{input_format}"
    logger.info(f"Loading dataset path={path}")
    xs = fs.glob(path, detail=False)
    ys = [hf_file_to_url(x, prefix=f"datasets/{dataset_name}") for x in xs]
    if not output_file:
        return ys

    write_file(ys, output_file=output_file, **kwargs)


def hf_file_to_url(line, prefix, base=None):
    return re.sub(
        f"^{prefix}/",
        f"https://huggingface.co/{prefix}/resolve/main/",
        line,
    )


def load_file_list(input_path: str, input_format: str):
    """
    Loads a list of file paths from a file when the input format specifies a list.

    This function checks if the provided `input_format` string starts with `"list:"`. If it does,
    it treats `input_path` as the path to a file containing a list of file paths (one per line).
    It reads this file, splits its contents into lines to create a list of file paths, and returns
    this list along with an optional format specifier extracted from `input_format`.

    Parameters:
        input_path (str): The path to the file containing the list of file paths.
        input_format (str): A string indicating the format of the input. It must start with
            `"list:"`. Optionally, it can be followed by a specific format after the colon
            (e.g., `"list:json"`).

    Returns:
        tuple or None:
            - If `input_format` starts with `"list:"`, returns a tuple `(file_list, format)`, where:
                - `file_list` (list): A list of file paths read from `input_path`.
                - `format` (str or None): The optional format specified after `"list:"` in `input_format`.
            - If `input_format` does not start with `"list:"`, returns `None`.

    Raises:
        ValueError: If `input_format` starts with `"list:"` but contains more than one colon,
            indicating an invalid format.

    Example:
        If `input_format` is `"list:json"` and `input_path` is `"files.txt"`, where `"files.txt"`
        contains:
            /path/to/file1.json
            /path/to/file2.json
        Then `load_file_list("files.txt", "list:json")` will return:
            (['/path/to/file1.json', '/path/to/file2.json'], 'json')
    """

    def file_list():
        if not input_format.startswith("list:"):
            return None, None
        xs = input_format.split(":")
        if len(xs) == 2:
            fmt = xs[1].strip()
            return input_path, fmt
        else:
            raise ValueError(f"expected format list:<format>, found {input_format}")

    fname, input_format = file_list()
    logger.info(f"Loading file list from {input_path} for {input_format}")
    if not fname:
        return None

    return read_file(fname, split_lines=True), input_format


def load_many(input_files: list, input_format: str = None, **kwargs):
    xs = (load_data(f, input_format, **kwargs) for f in input_files)
    return ds.concatenate_datasets(list(tqdm(xs)))


def log_samples(data, training_args: TrainingArguments, count=5):
    if training_args.local_rank == 0:
        if isinstance(data, ds.DatasetDict):
            if "train" not in data:
                logger.warning(f"could not log as train split is absent")
                return
            data = data["train"]

        if count < len(data):
            data = data.select(range(count))

        data = data.to_pandas()
        logger.info(f"{data}")


def load_data(
    input_path: str,
    input_format: str = "",
    remove_wrapper=True,
    max_records: int = None,
    **kwargs,
) -> ty.Union[ds.Dataset, ds.DatasetDict]:

    if remove_wrapper:
        data = load_data(input_path, input_format, remove_wrapper=False, **kwargs)
        if isinstance(data, ds.DatasetDict) and len(data) == 1:
            data = list(data.values())[0]

        if max_records and max_records < len(data):
            data = data.select(range(max_records))

        return data

    is_file = (
        input_path.startswith("gs://")
        or input_path.startswith("/")
        or input_path.startswith("./")
        or input_path.startswith("https://")
    )

    if input_format.startswith("list:"):
        input_files, input_format = load_file_list(input_path, input_format)
        return load_many(input_files, input_format, **kwargs)
    elif input_format == "arrow":
        logger.info(f"guessing", fmt="arrow")
        return ds.load_from_disk(input_path)
    elif is_file:
        logger.info(f"guessing format local file")
        if os.path.isdir(input_path):
            return ds.load_dataset(input_format, data_dir=input_path, **kwargs)
        else:
            return ds.load_dataset(input_format, data_files=input_path, **kwargs)
    else:
        logger.info(f"loading", fmt="hugging_face")
        return ds.load_dataset(input_path, **kwargs)


def save_data(
    data: ds.Dataset,
    output_path: str,
    output_format: str,
    overwrite=False,
    create_parents=True,
    **kwargs,
) -> str:

    if os.path.isdir(output_path):
        output_path = f"{output_path}/0.parquet"

    if create_parents and (not os.path.exists(os.path.dirname(output_path))):
        os.makedirs(os.path.dirname(output_path))

    if not overwrite and os.path.exists(output_path):
        raise FileExistsError(
            f"{output_path} already exists. set overwrite=True to overwrite."
        )

    if output_format == "parquet":
        data.to_parquet(output_path, **kwargs)
    elif output_format == "json":
        data.to_json(output_path, **kwargs)
    else:
        raise ValueError(f"unsupported output_format: {output_format}")

    return output_path


def save_dict(data: ds.DatasetDict, output_path: str, file_name: str, **kwargs):
    for split, it in tqdm(data.items()):
        path = os.path.join(output_path, split, file_name)
        save_data(it, path, **kwargs)


if __name__ == "__main__":
    fire.Fire()
