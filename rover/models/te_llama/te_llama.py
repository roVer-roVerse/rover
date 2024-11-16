# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import io
import os
import re
import gc
from contextlib import contextmanager

import fire
from loguru import logger
import torch
from torch import nn

import transformer_engine as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformer_engine.pytorch.fp8 import fp8_model_init

import transformers
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaConfig,
)
from transformers.modeling_utils import (
    _add_variant,
    load_state_dict,
    _load_state_dict_into_model,
)
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files

from rover.models.titan.layers.decoder_te import remove_extra_state


@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = (
        transformers.models.llama.modeling_llama.LlamaDecoderLayer
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = (
            original_llama_decoder_cls
        )


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `LlamaDecoderLayer` and easier to replace it in the code.

    Args:
        config: LlamaConfig
        args: positional args (for compatibility with `LlamaDecoderLayer`)
        kwargs: keyword args (for compatibility with `LlamaDecoderLayer`)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
        )
        te_rope = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads
        )
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, *args, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return (
            super().forward(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=self.te_rope_emb,
            ),
        )


def to_hf(te, hf, config):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in te.keys():
        layer_prefix_pat = r"model.layers.\d+."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    weight_tie = True
    common_layers = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        # todo: this is absent from the output due to weight tieing
        # "lm_head.weight",
    ]

    for layer_prefix in common_layers:
        hf[layer_prefix] = te[layer_prefix].data[:]

    if weight_tie:
        hf["lm_head.weight"] = te["model.embed_tokens.weight"].data[:]

    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        src = "self_attention.layernorm_qkv.layer_norm_weight"
        dst = "input_layernorm.weight"
        if layer_prefix + src in te:
            hf[layer_prefix + dst] = te[layer_prefix + src].data[:]

        src = "self_attention.layernorm_qkv.query_weight"
        dst = "self_attn.q_proj.weight"
        if layer_prefix + src in te:
            hf[layer_prefix + dst] = te[layer_prefix + src].data[:]

        src = "self_attention.layernorm_qkv.key_weight"
        dst = "self_attn.k_proj.weight"

        if layer_prefix + src in te:
            hf[layer_prefix + dst] = te[layer_prefix + src].data[:]

        src = "self_attention.layernorm_qkv.value_weight"
        dst = "self_attn.v_proj.weight"

        if layer_prefix + src in te:
            hf[layer_prefix + dst] = te[layer_prefix + src].data[:]

        src = "self_attention.proj.weight"
        dst = "self_attn.o_proj.weight"

        if layer_prefix + src in te:
            hf[layer_prefix + dst] = te[layer_prefix + src].data[:]

        src = "layernorm_mlp.layer_norm_weight"
        dst = "post_attention_layernorm.weight"

        if layer_prefix + src in te:
            hf[layer_prefix + dst] = te[layer_prefix + src].data[:]

        # It may happen that gate_proj.weight and up_proj.weight will be in the different files, so we need to
        # load them separately.

        src = "layernorm_mlp.fc1_weight"
        dst = "mlp.gate_proj.weight"
        dst2 = "mlp.up_proj.weight"

        if layer_prefix + src in te:
            hf[layer_prefix + dst] = te[layer_prefix + src].data[
                : config.intermediate_size
            ]
            hf[layer_prefix + dst2] = te[layer_prefix + src].data[
                config.intermediate_size :
            ]

        src = "layernorm_mlp.fc2_weight"
        dst = "mlp.down_proj.weight"

        if layer_prefix + src in te:
            hf[layer_prefix + dst] = te[layer_prefix + src].data[
                : config.intermediate_size
            ]

    return hf


def save_model(te_model, path):
    te_state = te_model.state_dict()
    hf_state = to_hf(te_state, {}, te_model.config)
    os.makedirs(path, exist_ok=False)
    torch.save(hf_state, f"{path}/pytorch_model.bin")
    te_model.config.save_pretrained(path)
    te_model.generation_config.save_pretrained(path)


from safetensors.torch import load_file
import subprocess as sp


def te_to_hf(input_path, output_path):
    te_state = load_file(f"{input_path}/model.safetensors")
    conf = transformers.AutoConfig.from_pretrained(input_path)
    hf_state = to_hf(te_state, {}, conf)
    os.makedirs(output_path, exist_ok=False)
    torch.save(hf_state, f"{output_path}/pytorch_model.bin")
    conf.save_pretrained(output_path)
    sp.run(f"cp {input_path}/*.json  {output_path}/", shell=True, check=True)


class TELlamaForCausalLM:
    """
    Causal LM created with `LlamaModel`. The underlying `LlamaDecoderLayer`
    class is monkey-patched with `TELlamaDecoderLayer` class before
    initializing the causal LM with `LlamaForCausalLM`.

    Args:
        config: LlamaConfig
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            llama_for_causal_lm = LlamaForCausalLM(config)
            llama_for_causal_lm._register_state_dict_hook(remove_extra_state)
        return llama_for_causal_lm

    @classmethod
    def from_pretrained_local(
        cls, pretrained_model_name_or_path, *args, config, **kwargs
    ):
        """
        Custom method adapted from `from_pretrained` method in HuggingFace
        Transformers repo: https://github.com/huggingface/transformers/blob/f497f564bb76697edab09184a252fc1b1a326d1e/src/transformers/modeling_utils.py#L2579
        """
        # Before loading the model, set the default dtype for torch
        torch.set_default_dtype(kwargs["torch_dtype"])

        # Load the vanilla model weights
        vanilla_model = cls(config)
        subfolder = ""
        variant = None
        if os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
        ):
            # Load from a sharded PyTorch checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
            is_sharded = True
        elif os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
            )
        ):
            # Load from a sharded PyTorch checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
            )
            is_sharded = True
        else:
            raise AssertionError(
                "Only sharded PyTorch ckpt format supported at the moment"
            )

        resolved_archive_file, _ = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            archive_file,
        )

        # If the checkpoint is not sharded, it's a trivial sharding case
        if not is_sharded:
            assert not isinstance(resolved_archive_file, list)
            resolved_archive_file = [resolved_archive_file]

        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)
            # replace_params copies parameters relevant only to TransformerEngine
            replace_params(state_dict, vanilla_model.state_dict(), config)
            # _load_state_dict_into_model copies parameters other than those in TransformerEngine
            _load_state_dict_into_model(vanilla_model, state_dict, start_prefix="")

            # Force mem release. Taken from huggingface code
            del state_dict
            gc.collect()

        return vanilla_model


def replace_params(hf_state_dict, te_state_dict, config):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = r"model.layers.\d+."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    common_layers = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for layer_prefix in common_layers:
        te_state_dict[layer_prefix].data[:] = hf_state_dict[layer_prefix].data[:]

    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        if layer_prefix + "input_layernorm.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"
            ].data[:] = hf_state_dict[layer_prefix + "input_layernorm.weight"].data[:]

        if layer_prefix + "self_attn.q_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.query_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.q_proj.weight"].data[:]

        if layer_prefix + "self_attn.k_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.key_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.k_proj.weight"].data[:]

        if layer_prefix + "self_attn.v_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.value_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.v_proj.weight"].data[:]

        if layer_prefix + "self_attn.o_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.proj.weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.o_proj.weight"].data[:]
            )

        if layer_prefix + "post_attention_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.layer_norm_weight"].data[:] = (
                hf_state_dict[layer_prefix + "post_attention_layernorm.weight"].data[:]
            )

        # It may happen that gate_proj.weight and up_proj.weight will be in the different files, so we need to
        # load them separately.
        if layer_prefix + "mlp.gate_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                : config.intermediate_size
            ] = hf_state_dict[layer_prefix + "mlp.gate_proj.weight"].data

        if layer_prefix + "mlp.up_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                config.intermediate_size :
            ] = hf_state_dict[layer_prefix + "mlp.up_proj.weight"].data

        if layer_prefix + "mlp.down_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc2_weight"].data[:] = (
                hf_state_dict[layer_prefix + "mlp.down_proj.weight"].data[:]
            )
    return all_layer_prefixes


if __name__ == "__main__":
    fire.Fire()