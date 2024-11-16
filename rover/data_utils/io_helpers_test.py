import os
import tempfile
import unittest

from rover.data_utils.io_helpers import write_many, write_file, read_file


class TestWrite(unittest.TestCase):
    def test_write_simple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            input_files = ["input1.parquet", "input2.parquet"]
            write_file(input_files, input_path)
            files_written = read_file(input_path, split_lines=True)
            self.assertListEqual(files_written, input_files)


class TestWriteMany(unittest.TestCase):
    """
    https://chatgpt.com/share/671f8cd9-e61c-8003-97ac-8c9ff3245a0e
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_many_with_integer_file_numbers(self):
        data = ["data1", "data2", "data3"]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(data, file_numbers=3, output_regex=output_regex)

        for i in range(3):
            file_path = output_regex.format(i=i)
            with open(file_path, "r") as f:
                self.assertEqual(f.read(), data[i])

    def test_write_many_with_iterable_file_numbers(self):
        data = ["data1", "data2", "data3"]
        file_numbers = [0, 1, 2]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(data, file_numbers=file_numbers, output_regex=output_regex)

        for i in file_numbers:
            file_path = output_regex.format(i=i)
            with open(file_path, "r") as f:
                self.assertEqual(f.read(), data[i])

    def test_write_many_with_tuple_file_numbers(self):
        data = ["data1", "data2", "data3"]
        file_numbers = (0, 3)
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(data, file_numbers=file_numbers, output_regex=output_regex)

        for i in range(3):
            file_path = output_regex.format(i=i)
            with open(file_path, "r") as f:
                self.assertEqual(f.read(), data[i])

    def test_output_regex_formatting(self):
        data = ["data1", "data2"]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(data, file_numbers=2, output_regex=output_regex)

        for i in range(2):
            file_path = output_regex.format(i=i)
            with open(file_path, "r") as f:
                self.assertEqual(f.read(), data[i])

    def test_custom_writer_function(self):
        def custom_writer(data, output_file, **kwargs):
            with open(output_file, "w") as f:
                f.write(f"custom: {data}")

        data = ["data1", "data2"]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(
            data, file_numbers=2, output_regex=output_regex, writer=custom_writer
        )

        for i in range(2):
            file_path = output_regex.format(i=i)
            with open(file_path, "r") as f:
                self.assertEqual(f.read(), f"custom: {data[i]}")

    def test_default_writer_function(self):
        data = ["data1"]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(data, file_numbers=1, output_regex=output_regex)

        file_path = output_regex.format(i=0)
        with open(file_path, "r") as f:
            self.assertEqual(f.read(), data[0])

    def test_data_length_mismatch_with_file_numbers(self):
        data = ["data1", "data2"]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(data, file_numbers=3, output_regex=output_regex)

        for i in range(2):
            file_path = output_regex.format(i=i)
            with open(file_path, "r") as f:
                self.assertEqual(f.read(), data[i])

    def test_kwargs_passed_to_writer(self):
        def custom_writer(data, output_file, mode):
            with open(output_file, mode) as f:
                f.write(data)

        data = ["data1"]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(
            data,
            file_numbers=1,
            output_regex=output_regex,
            writer=custom_writer,
            mode="w",
        )

        file_path = output_regex.format(i=0)
        with open(file_path, "r") as f:
            self.assertEqual(f.read(), data[0])

    def test_with_empty_data(self):
        data = []
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(data, file_numbers=0, output_regex=output_regex)

        self.assertEqual(len(os.listdir(self.temp_dir.name)), 0)

    def test_large_data(self):
        data = [f"data{i}" for i in range(1000)]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        write_many(data, file_numbers=1000, output_regex=output_regex)

        for i in range(1000):
            file_path = output_regex.format(i=i)
            with open(file_path, "r") as f:
                self.assertEqual(f.read(), data[i])

    def test_incorrect_type_for_file_numbers(self):
        data = ["data1"]
        output_regex = os.path.join(self.temp_dir.name, "output_{i}.txt")
        with self.assertRaises(TypeError):
            write_many(data, file_numbers="invalid", output_regex=output_regex)

    def test_output_regex_missing_placeholder(self):
        data = ["data1"]
        output_regex = os.path.join(self.temp_dir.name, "output.txt")
        write_many(data, file_numbers=1, output_regex=output_regex)

        file_path = output_regex
        with open(file_path, "r") as f:
            self.assertEqual(f.read(), data[0])
