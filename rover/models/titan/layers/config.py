from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    attn_dim: int = 64
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    ffn_dim: Optional[int] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    norm_type: str = "rmsnorm"

    decoder_type: str = "default"

    def __post_init__(self):
        if self.ffn_dim_multiplier is not None:
            self.ffn_dim = int(self.ffn_dim_multiplier * self.dim)
            self.ffn_dim = self.multiple_of * (
                (self.ffn_dim + self.multiple_of - 1) // self.multiple_of
            )

        assert self.n_heads * self.attn_dim == self.dim

        assert (
            self.max_seq_len % 16 == 0
        ), "for TransformerEngine, max_seq_len must be divisible by 16"


llama3_configs = {
    "debugmodel": ModelArgs(
        dim=128, n_layers=2, n_heads=4, attn_dim=32, rope_theta=500000
    ),
    "125M": ModelArgs(
        dim=64 * 12,
        n_layers=32,
        n_heads=12,
        n_kv_heads=3,
        ffn_dim_multiplier=1.3,
        rope_theta=500000,
    ),
    "400M": ModelArgs(
        dim=1024,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        rope_theta=500000,
    ),
    "1B": ModelArgs(
        dim=2048,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        rope_theta=500000,
    ),
}
