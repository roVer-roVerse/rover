import unittest

import datasets as ds

from rover.tokens.constant_length import packed_dataset


class TestPackedCollator(unittest.TestCase):

    def short_seq_dataset(self):
        """Fixture that provides a dataset for short sequence length tests."""
        test1 = dict(
            bos_token_id=2,
            eos_token_id=3,
            seq_length=4,
            input_ids=[
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203],
                [300, 301, 302],
            ],
            expected=[
                [2, 100, 101, 102],
                [2, 103, 104, 3],
                [2, 200, 201, 202],
                [2, 203, 3, 2],
                [2, 300, 301, 302],
            ],
        )
        return [test1]

    def test_packed_collator_short_seq_length(self):
        """Test the collator with a sequence length shorter than the default."""
        recs = self.short_seq_dataset()
        for rec in recs:
            result = list(
                packed_dataset(
                    ds.Dataset.from_dict(dict(input_ids=rec["input_ids"])),
                    bos_token_id=rec["bos_token_id"],
                    eos_token_id=rec["eos_token_id"],
                    seq_length=rec["seq_length"],
                )
            )
            output_ids = [xs["input_ids"].tolist() for xs in result]
            self.assertListEqual(output_ids, rec["expected"])
