#!/usr/bin/env python
# coding=utf-8
import os

import datasets as ds
import transformers as tr
from loguru import logger

from rover.data_utils.chat_templates.collator_helpers import (
    PadWithLabelsCollator,
)
from rover.workflows.pretrain.model_factory import save_model


def train3(
    model: tr.PreTrainedModel,
    raw_datasets: ds.DatasetDict,
    training_args: tr.TrainingArguments,
    tokenizer: tr.PreTrainedTokenizer,
    max_length,
    resume_from_checkpoint=None,
):

    tr.set_seed(42)

    if "text" in raw_datasets.column_names:
        raise ValueError(f"Not sure if to tokenize if input already contains text")

    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    train_data, eval_data = None, None
    if training_args.do_train:
        train_data = raw_datasets["train"]
        # train_data = train_data.map(
        #     lambda rec: pad_with_labels(rec, tokenizer, max_length),
        #     batched=True,
        #     batch_size=training_args.per_gpu_train_batch_size,
        #     remove_columns=train_data.column_names,
        # )

    if training_args.do_eval:
        eval_data = raw_datasets["validation"]
        # eval_data = eval_data.map(
        #     lambda rec: pad_with_labels(rec, tokenizer, max_length),
        #     batched=True,
        #     batch_size=training_args.per_gpu_eval_batch_size,
        #     remove_columns=eval_data.column_names,
        # )

    collator = PadWithLabelsCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    trainer = tr.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        # max_seq_length=training_args.max_seq_length,
        # tokenizer=tokenizer,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_data)

        save_model(trainer, metrics, split="train", save_trainer_state=True)
        # todo: create model card and dump
        # datasets, hypterparameters

    if "test" in raw_datasets:
        metrics = trainer.evaluate(raw_datasets["test"])
        metrics["test_samples"] = len(raw_datasets["test"])
        save_model(trainer, metrics, split="test")

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    logger.info("done")
