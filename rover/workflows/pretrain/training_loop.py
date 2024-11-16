import datasets as ds
import transformers as tr
from loguru import logger
from transformers.integrations import TensorBoardCallback

from rover.data_utils.mixer import DataResult
from rover.workflows.pretrain.callback_helpers import PerplexityCB
from rover.workflows.pretrain.model_factory import save_model


def train(
    data: DataResult,
    model,
    tokenizer,
    training_args: tr.Seq2SeqTrainingArguments,
    device: str,
):
    # data_collator = tr.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model.to(device=device)
    model.train()

    trainer = tr.Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data.collator,
        train_dataset=data.data["train"],
        eval_dataset=data.data.get("validation"),
        callbacks=[PerplexityCB(), TensorBoardCallback],
    )

    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    save_model(trainer)

    logger.info(train_result)
    return train_result
