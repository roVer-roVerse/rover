import fire
import os
import torch
import typing as ty
import numpy as np
from loguru import logger
import pandas as pd
import transformers as tr
import datasets as ds

from subprocess import run
import sys


class PerplexityCB(tr.TrainerCallback):
    def __init__(self, min_ppl=3.0):
        self.min_ppl = np.exp(min_ppl)
        self.last_ppl = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if "loss" not in logs:
            return

        ppl = np.exp(logs["loss"])
        delta_ppl = self.last_ppl - ppl
        self.last_ppl = ppl

        # avoid logging very large values
        if ppl < self.min_ppl:
            logs["perplexity"] = ppl
            logs["delta_ppl"] = delta_ppl
