import torch
import unittest
from rover.models.titan.llama.model import TritonLlama3
from rover.models.titan.layers.config import ModelArgs


def loss_fn(pred, labels, reduction="sum"):
    input = pred.flatten(0, 1)
    target = labels.flatten(0, 1)
    return torch.nn.functional.cross_entropy(input.float(), target, reduction=reduction)


class ModelTest(unittest.TestCase):
    def test_init_weights(self):
        conf = ModelArgs(
            dim=256,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            ffn_dim_multiplier=1.3,
            vocab_size=1024,
            max_seq_len=32,
        )
        with torch.device("cpu"):
            model = TritonLlama3(conf)

        model.train()

        batch_size = 5
        input_ids = torch.randint(
            0, conf.vocab_size, [batch_size, conf.max_seq_len], dtype=torch.long
        )
        labels = input_ids.clone()

        logits = model(input_ids)
        self.assertEqual(
            logits.shape, torch.Size([batch_size, conf.max_seq_len, conf.vocab_size])
        )
        loss = loss_fn(logits, labels) / labels.numel()
        loss2 = loss_fn(logits, labels, reduction="mean")

        delta = loss - loss2
        self.assertTrue(torch.abs(delta) < torch.FloatTensor([1e-5]))

        loss.backward()
