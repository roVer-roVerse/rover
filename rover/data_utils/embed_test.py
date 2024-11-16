import os

import fire

from rover.data_utils.embed import embed
from subprocess import run


def tiny():
    base = "./out/datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2"
    tb_path = f"{base}-embed/tb2/"
    run(f"rm -rf {tb_path}", shell=True, check=True)
    embed(
        model_path="TaylorAI/bge-micro",
        input_path=f"{base}/*.parquet",
        output_path=f"{base}-embed/10k.parquet",
        tensorboard_path=tb_path,
        tensorboard_labels=["audience", "format", "seed_data", "text"],
        max_records=10000,
        overwrite=True,
    )

    print(f"tensorboard --logdir {tb_path}")


if __name__ == "__main__":
    fire.Fire()
