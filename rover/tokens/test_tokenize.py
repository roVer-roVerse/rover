import unittest
import tempfile
import os
import datasets
import transformers as tr
import datasets as ds

from rover.data_utils.io_helpers import write_file, read_file
from rover.tokens.token_main import tokenize


class TestTokenize(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_path = "meta-llama/Llama-3.2-1B"  # Or any other pre-trained tokenizer

    def test_tokenize_basic(self):
        # Create temporary directories and files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Paths for input and output files
            input_path = os.path.join(tmpdir, "input.parquet")
            output_path = os.path.join(tmpdir, "output.parquet")

            # Create small data_utils and save as parquet
            data_dict = {
                "text": [
                    "Hello, how are you?",
                    "This is a test.",
                    "Another test sentence.",
                ]
            }
            data = datasets.Dataset.from_dict(data_dict)
            data.to_parquet(input_path)

            # Call the tokenize function
            tokenize(
                tokenizer_name_path=self.name_path,
                input_path=input_path,
                output_path=output_path,
                input_format="parquet",
                output_format="parquet",
                add_special_tokens=False,
                batch_size=1024,
            )

            # Read the output file
            output_data = datasets.Dataset.from_parquet(output_path)

            # Load the tokenizer
            tokenizer = tr.AutoTokenizer.from_pretrained(self.name_path)

            # Validate each row by decoding and compare to original
            for original_text, encoded_ids in zip(
                data["text"], output_data["input_ids"]
            ):
                decoded_text = tokenizer.decode(encoded_ids, skip_special_tokens=True)
                self.assertEqual(decoded_text.strip(), original_text.strip())

    def test_multi_file(self):
        # Create temporary directories and files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Paths for input and output files

            input_path1 = os.path.join(tmpdir, "input1.parquet")
            input_path2 = os.path.join(tmpdir, "input2.parquet")

            output_path = os.path.join(tmpdir, "output.parquet")

            sentences = [
                f"This is sentence number {i}." for i in range(20)
            ]  # 20 sentences

            first_file_rows = 12
            data1 = datasets.Dataset.from_dict({"prompt": sentences[:first_file_rows]})
            data1.to_parquet(input_path1)
            data2 = datasets.Dataset.from_dict({"prompt": sentences[first_file_rows:]})
            data2.to_parquet(input_path2)

            input_path = os.path.join(tmpdir, "input.txt")
            input_files = [input_path1, input_path2]
            write_file(input_files, input_path)
            files_written = read_file(input_path, split_lines=True)
            self.assertListEqual(files_written, input_files)

            tokenize(
                tokenizer_name_path=self.name_path,
                input_path=input_path,
                input_format="list:parquet",
                output_path=output_path,
                column_name="prompt",
                output_column_name=None,
                output_format="parquet",
                add_special_tokens=True,
                batch_size=4,
            )

            # Read the output file
            output_data = datasets.Dataset.from_parquet(output_path)

            # Load the tokenizer
            tokenizer = tr.AutoTokenizer.from_pretrained(self.name_path)

            # Validate each row by decoding and compare to original
            for original_text, encoded_ids in zip(
                sentences, output_data["prompt_input_ids"]
            ):
                decoded_text = tokenizer.decode(encoded_ids, skip_special_tokens=True)
                self.assertEqual(decoded_text.strip(), original_text.strip())

    # Repeat similar test with special_tokens=True, batch_size=4, and input is larger
    def test_tokenize_special_tokens_batch(self):
        # Create temporary directories and files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Paths for input and output files
            input_path = os.path.join(tmpdir, "input.parquet")
            output_path = os.path.join(tmpdir, "output.parquet")

            # Create larger data_utils
            sentences = [
                f"This is sentence number {i}." for i in range(20)
            ]  # 20 sentences
            data_dict = {"prompt": sentences}
            data = datasets.Dataset.from_dict(data_dict)
            data.to_parquet(input_path)

            # Call the tokenize function with special_tokens=True, batch_size=4
            tokenize(
                tokenizer_name_path=self.name_path,
                input_path=input_path,
                output_path=output_path,
                column_name="prompt",
                output_column_name=None,
                input_format="parquet",
                output_format="parquet",
                add_special_tokens=True,
                batch_size=4,
            )

            # Read the output file
            output_data = datasets.Dataset.from_parquet(output_path)

            # Load the tokenizer
            tokenizer = tr.AutoTokenizer.from_pretrained(self.name_path)

            # Validate each row by decoding and compare to original
            for original_text, encoded_ids in zip(
                data["prompt"], output_data["prompt_input_ids"]
            ):
                decoded_text = tokenizer.decode(encoded_ids, skip_special_tokens=True)
                self.assertEqual(decoded_text.strip(), original_text.strip())
