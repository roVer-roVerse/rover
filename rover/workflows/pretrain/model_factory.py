from dataclasses import dataclass, field
import transformers as tr

from loguru import logger

import fire


@dataclass
class ModelArguments:
    model_name_or_path_or_config: str
    tokenizer_name: str = field(default=None)
    te: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")
    dtype: str = field(default="bfloat16")

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path_or_config


def load_model(c: ModelArguments, kv_cache=False):
    if c.te:
        from rover.models.te_llama.utils import load_te_from_hf

        framework = "te"
        model = load_te_from_hf(c.model_name_or_path_or_config)
    else:
        framework = "hf"
        model = tr.AutoModelForCausalLM.from_pretrained(
            c.model_name_or_path_or_config,
            attn_implementation=c.attn_implementation,
            torch_dtype=c.dtype,
        )

    logger.info(f"model framework={framework}")
    logger.info(f"{model}")
    model.config.use_cache = kv_cache
    return model


def load_tokenizer(tokenizer_name, trust_remote_code=False, add_pad_token=True):
    tokenizer = tr.AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )

    if add_pad_token and (not tokenizer.pad_token):
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def from_config(
    path: str,
    *,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    output_dir: str = None,
):
    conf = tr.AutoConfig.from_pretrained(path)
    model = tr.AutoModel.from_config(conf)
    if output_dir:
        tok = load_tokenizer(tokenizer_name)
        model.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)
        return

    return model


def save_model(
    trainer: tr.Trainer,
    metrics=None,
    kv_cache=True,
    split="train",
    save_trainer_state=False,
):
    trainer.save_model()
    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = kv_cache
        trainer.model.config.save_pretrained(trainer.args.output_dir)
        trainer.tokenizer.save_pretrained(trainer.args.output_dir)
        if metrics:
            trainer.log_metrics(split, metrics)
            trainer.save_metrics(split, metrics)

        if save_trainer_state:
            trainer.save_state()


if __name__ == "__main__":
    fire.Fire()
