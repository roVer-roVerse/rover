import json
import os
import typing as ty

from loguru import logger
from tqdm import tqdm


def write_file(
    lines: ty.Union[str, ty.Iterable],
    output_file: str,
    create_parents=True,
    overwrite=False,
    to_json=False,
):

    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(output_file)

    if create_parents:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if isinstance(lines, str):
        lines = [lines]

    is_json = output_file.endswith(".json") or output_file.endswith(".jsonl")
    if is_json and (not to_json):
        logger.warning("if file ends with json, should enable to_json=True")

    logger.info(f"writing lines to {output_file}")
    with open(output_file, "w") as f:
        first = True
        for line in tqdm(lines):
            if not first:
                f.write("\n")
            if to_json:
                line = json.dumps(line)
            f.write(line)
            first = False


def write_many(
    data: ty.Iterable,
    file_numbers: ty.Union[int, ty.Iterable],
    output_regex: str,
    writer=None,
    **kwargs,
):

    if writer is None:
        writer = write_file

    if isinstance(file_numbers, int):
        file_numbers = range(file_numbers)
    elif isinstance(file_numbers, tuple):
        file_numbers = range(*file_numbers)
    elif isinstance(file_numbers, str):
        raise TypeError("file_numbers cannot be string")
    elif not isinstance(file_numbers, ty.Iterable):
        raise TypeError("file_numbers must be int or tuple")

    for i, d in tqdm(zip(file_numbers, data)):
        output_file = output_regex.format(i=i)
        writer(d, output_file=output_file, **kwargs)


def read_file(p: str, split_lines=False, strip=True):
    with open(p, "r") as f:
        if split_lines:
            xs = f.readlines()
            if strip:
                xs = [x.strip() for x in xs]

            assert isinstance(xs, list)
            return xs
        else:
            xs = f.readline()
            if strip:
                xs = xs.strip()
            return xs


def human_readable(n):
    ranges = [
        (1e18, "E"),
        (1e15, "P"),
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "u"),
    ]
    for i, (base, base_str) in enumerate(ranges[1:]):
        if abs(n) > base:
            out = f"{n/base:.2f}{base_str}"
            return out

    return f"{n:0.2e}"
