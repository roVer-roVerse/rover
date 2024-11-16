import os
import typing

import fire
import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from transformers.pipelines.pt_utils import KeyPairDataset, KeyDataset

from rover.data_utils.loader import load_data, save_data
import transformers as tr
import numpy as np
import datasets as ds
import torch.utils.tensorboard as tb


# todo: make this faster
def embed(
    model_path,
    input_path: str,
    output_path: str,
    *,
    input_format="parquet",
    output_format="parquet",
    column_name="text",
    output_column_name="embedding",
    overwrite=False,
    max_records=None,
    batch_size=128,
    tensorboard_path=None,
    tensorboard_labels=None,
    trust_remote_code=False,
):

    model = SentenceTransformer(model_path, trust_remote_code=trust_remote_code)
    data = load_data(
        input_path,
        input_format=input_format,
        remove_wrapper=True,
        max_records=max_records,
    )
    logger.info(f"{data}")
    logger.info(f"{data.select(range(min(5, len(data)))).to_pandas()}")
    if column_name not in data.column_names:
        logger.warning(f"{column_name} not found in {data.column_names}")

    def foo(it):
        es = model.encode(it, normalize=True, batch_size=batch_size)
        return {output_column_name: es}

    data = data.map(
        foo, batched=True, batch_size=batch_size * 8, input_columns=column_name
    )
    if output_path:
        save_data(data, output_path, output_format, overwrite=overwrite)

    if tensorboard_path:
        push_tensorboard(
            data,
            output_path=tensorboard_path,
            column_name=output_column_name,
            labels=tensorboard_labels,
            overwrite=overwrite,
        )
    return data


def push_tensorboard(
    input_path: typing.Union[str, ds.Dataset],
    output_path: str,
    *,
    input_format="parquet",
    column_name="embedding",
    labels: list = None,
    overwrite=False,
    max_records: int = None,
):

    if not overwrite and os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists")

    writer = tb.SummaryWriter(output_path)

    if not isinstance(input_path, ds.Dataset):
        data = load_data(
            input_path,
            input_format=input_format,
            remove_wrapper=True,
            max_records=max_records,
        )
    else:
        data = input_path

    data.set_format("np", columns=[column_name])
    features = data[column_name]

    metadata, metadata_header = None, None
    if labels:

        def sanitize(rec):
            return {label: tsv_safe(rec[label]) for label in labels}

        metadata = data.select_columns(labels)
        metadata = metadata.map(
            sanitize,
            batched=False,
            # num_proc=os.cpu_count(),
            desc="removing whitespaces for tsv",
            cache_file_name=None,
        )
        metadata = metadata.to_pandas()
        metadata_header = metadata.columns.to_list()
        metadata = list(metadata.itertuples(index=False))

    writer.add_embedding(features, metadata=metadata, metadata_header=metadata_header)
    writer.close()


def tsv_safe(m: str):
    m = m.replace("\t", "    ")
    m = m.replace("\n", "<br>")
    return m


#
# def embed_hf(
#     model_path,
#     input_path: str,
#     output_path: str,
#     *,
#     input_format="parquet",
#     output_format="parquet",
#     column_name="text",
#     output_column_name="embedding",
#     overwrite=False,
#     batch_size=128,
#     max_records: int = None,
# ):
#     pipe = tr.pipeline("feature-extraction", model=model_path)
#     data_utils = load_data(input_path, input_format=input_format, remove_wrapper=True)#
#     logger.info(f"{data_utils}")
#     logger.info(f"{data_utils.select(range(5)).to_pandas()}")
#     if column_name not in data_utils.column_names:
#         logger.warning(f"{column_name} not found in {data_utils.column_names}")
#
#     es = pipe(KeyDataset(data_utils, column_name), num_workers=4, batch_size=batch_size)
#     data_utils = data_utils.add_column(output_column_name, list(tqdm.tqdm(es, total=len(data_utils))))
#     if output_path:
#         save_data(data_utils, output_path, output_format, overwrite=overwrite)
#     else:
#         return data_utils


if __name__ == "__main__":
    fire.Fire()
