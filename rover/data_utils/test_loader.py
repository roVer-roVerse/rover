import os
import tempfile
import unittest
from unittest import TestCase

from rover.data_utils.loader import hfls, load_file_list


class TestHfLs(TestCase):
    def test_hfls(self):
        xs = hfls("openai/gsm8k", prefix="socratic")
        self.assertCountEqual(
            xs,
            [
                "https://huggingface.co/datasets/openai/gsm8k/resolve/main/socratic/test-00000-of-00001.parquet",
                "https://huggingface.co/datasets/openai/gsm8k/resolve/main/socratic/train-00000-of-00001.parquet",
            ],
        )


class TestLoadFileList(unittest.TestCase):
    def test_load_file_list(self):
        test_cases = [
            {
                "name": "# Positive Test Case 1",
                "input_path": None,  # Placeholder for temp file path
                "input_format": "list:json",
                "file_contents": "/path/to/file1.json\n/path/to/file2.json",
                "expected_output": (
                    ["/path/to/file1.json", "/path/to/file2.json"],
                    "json",
                ),
            },
            {
                "name": "# Positive Test Case 2",
                "input_path": None,
                "input_format": "list:",
                "file_contents": "/path/to/file1.txt\n/path/to/file2.txt",
                "expected_output": (["/path/to/file1.txt", "/path/to/file2.txt"], ""),
            },
            {
                "name": "# Positive Test Case 3",
                "input_path": None,
                "input_format": "list:csv",
                "file_contents": "",
                "expected_output": ([], "csv"),
            },
            {
                "name": "# Positive Test Case 4",
                "input_path": None,
                "input_format": "csv",
                "file_contents": None,
                "expected_output": None,
            },
            {
                "name": "# Positive Test Case 5",
                "input_path": None,
                "input_format": "list:json:extra",
                "file_contents": None,
                "expected_exception": ValueError,
                "expected_exception_msg": "expected format list:<format>, found list:json:extra",
            },
            {
                "name": "# Positive Test Case 6",
                "input_path": "nonexistent.txt",
                "input_format": "list:json",
                "file_contents": None,
                "expected_exception": FileNotFoundError,
            },
        ]

        for case in test_cases:
            input_path = case["input_path"]
            input_format = case["input_format"]
            file_contents = case["file_contents"]
            expected_output = case.get("expected_output")
            expected_exception = case.get("expected_exception")
            expected_exception_msg = case.get("expected_exception_msg")

            name = case["name"]
            with self.subTest(f"Test Case {name}"):
                if file_contents is not None:
                    with tempfile.NamedTemporaryFile(
                        mode="w+", delete=False
                    ) as temp_file:
                        temp_file.write(file_contents)
                        temp_file_name = temp_file.name
                    try:
                        input_path = temp_file_name
                        if expected_exception:
                            with self.assertRaises(expected_exception) as context:
                                load_file_list(input_path, input_format)
                            if expected_exception_msg:
                                self.assertEqual(
                                    str(context.exception), expected_exception_msg
                                )
                        else:
                            result = load_file_list(input_path, input_format)
                            self.assertEqual(result, expected_output)
                    finally:
                        os.remove(temp_file_name)
                else:
                    if expected_exception:
                        with self.assertRaises(expected_exception) as context:
                            load_file_list(input_path, input_format)
                        if expected_exception_msg:
                            self.assertEqual(
                                str(context.exception), expected_exception_msg
                            )
                    else:
                        result = load_file_list(input_path, input_format)
                        self.assertEqual(result, expected_output)
