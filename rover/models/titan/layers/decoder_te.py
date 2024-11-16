import io

import torch
from loguru import logger

import transformer_engine.pytorch as te

from rover.models.titan.layers.config import ModelArgs


class TEDecoderLayer(te.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to Titan's `DecoderLayer`
    """

    def __init__(self, config: ModelArgs):

        if config.norm_type == "rmsnorm":
            norm = "RMSNorm"
        elif config.norm_type == "layernorm":
            norm = "LayerNorm"
        else:
            raise ValueError(f"Unknown norm_type {config.norm_type}")

        super().__init__(
            hidden_size=config.dim,
            ffn_hidden_size=config.ffn_dim,
            num_attention_heads=config.n_heads,
            bias=False,
            layernorm_epsilon=config.norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization=norm,
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.n_kv_heads,
        )
        # te_rope = te.attention.RotaryPositionEmbedding(
        #     config.hidden_size // config.num_attention_heads
        # )
        # self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        xs = super().forward(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_cis,
        )
        return xs


def remove_extra_state(module, state, prefix, local_metadata):
    keys_to_delete = set()
    for param_key in state.keys():
        if isinstance(state[param_key], io.BytesIO):
            keys_to_delete.add(param_key)

    logger.warning(f"deleting keys count= {len(keys_to_delete)}")
    for k in keys_to_delete:
        del state[k]

    return state
