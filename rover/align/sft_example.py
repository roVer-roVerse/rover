import os

import fire
import transformers
from transformers import TrainingArguments

from rover.align.sft import train3
import rover.data_utils.chat_templates.chat_token as ct
from rover.data_utils.chat_templates.smol_ct import SmolLm2CT
from rover.data_utils.providers.popular import everyday_conversations
from rover.workflows.pretrain.model_factory import (
    load_model,
    ModelArguments,
    load_tokenizer,
)


def fetch_data(tokenizer, args: TrainingArguments, num_proc):
    with args.main_process_first(desc="Fetching data"):
        chat_tpl = ct.CTFactory(SmolLm2CT, tokenizer)
        everyday = everyday_conversations(
            chat_tpl, only_keep_output=True, num_proc=num_proc
        )

    return everyday


def main(model_name="HuggingFaceTB/SmolLM2-135M-Instruct", output_dir="out/tmp/sft"):

    # for rslora, alpha=8 should be fine
    # lora = peft.LoraConfig(r=256, lora_alpha=8, use_rslora=True)
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        report_to=[],
        dataloader_num_workers=os.cpu_count(),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        logging_steps=10,
        save_steps=100,
        do_eval=True,
        # eval_on_start=True,
        eval_steps=20,
        max_steps=200,
        eval_delay=50,
        eval_strategy="steps",
        bf16=True,
        # tf32=True,
        bf16_full_eval=True,
    )

    tokenizer = load_tokenizer(model_name, add_pad_token=True)
    data = fetch_data(tokenizer, args=args, num_proc=os.cpu_count())
    model = load_model(
        ModelArguments(
            model_name_or_path_or_config=model_name,
            attn_implementation="sdpa",
        ),
    )
    model.to("mps")
    model.train()
    train3(
        model=model,
        raw_datasets=data,
        training_args=args,
        tokenizer=tokenizer,
        max_length=2048,
    )


if __name__ == "__main__":
    fire.Fire(main)
