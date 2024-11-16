# import unittest
#
# import torch
#
# from rover.data_utils.chat_templates.collator_helpers import pad_with_labels
# import transformers as tr
#
#
# class TestCollatorHelpers(unittest.TestCase):
#     def test_pad_with_labels(self):
#         model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
#         out = pad_with_labels(
#             msg=dict(
#                 input_ids=[[1, 2, 3], [21, 22, 23, 24, 25, 26], [31, 32]],
#                 labels=[[1, 2, -100], [21, 22, 23, -24, -100, -100], [31, 32]],
#             ),
#             tokenizer=tr.AutoTokenizer.from_pretrained(model_name),
#             max_length=4,
#         )
#         expected = {
#             "input_ids": torch.tensor(
#                 [[1, 2, 3, 2, 2, 2], [21, 22, 23, 24, 25, 26], [31, 32, 2, 2, 2, 2]]
#             ),
#             "attention_mask": torch.tensor(
#                 [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]]
#             ),
#             "labels": torch.tensor(
#                 [
#                     [1, 2, -100, -100, -100, -100],
#                     [21, 22, 23, -24, -100, -100],
#                     [31, 32, -100, -100, -100, -100],
#                 ]
#             ),
#         }
#
#         for k in expected.keys():
#             delta = (out.data[k] - expected[k]).sum().tolist()
#             self.assertEqual(delta, 0, f"{k} mismatch")
