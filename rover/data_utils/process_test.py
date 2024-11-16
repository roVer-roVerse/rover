import unittest
from datasets import Dataset, DatasetDict

from rover.data_utils.process import group_by, bucket_by


class TestDatasetUtilities(unittest.TestCase):
    def setUp(self):
        """Set up test data for the test cases."""
        self.data = Dataset.from_dict(
            {"category": ["A", "B", "A", "C", "B"], "value": [1, 2, 3, 4, 5]}
        )

    def test_group_by(self):
        """Test the group_by method with valid input."""
        grouped = group_by(self.data, key="category")

        # Check if the result is a DatasetDict
        self.assertIsInstance(grouped, DatasetDict)

        # Verify the keys in the grouped DatasetDict
        self.assertEqual(set(grouped.keys()), {"A", "B", "C"})

        # Check the number of rows for each group
        self.assertEqual(len(grouped["A"]), 2)
        self.assertEqual(len(grouped["B"]), 2)
        self.assertEqual(len(grouped["C"]), 1)

    def test_bucket_by(self):
        """Test the bucket_by method with valid input."""
        # Using a small number of buckets for simplicity
        n_buckets = 2
        bucketed = bucket_by(self.data, key="category", n=n_buckets)

        # Check if the result is a DatasetDict
        self.assertIsInstance(bucketed, DatasetDict)

        # Verify that there are exactly `n_buckets` keys
        self.assertEqual(len(bucketed.keys()), n_buckets)

        # Check if the bucket IDs are strings (since we convert them to strings in the method)
        for bucket_id in bucketed.keys():
            self.assertIsInstance(bucket_id, str)

        # Verify that all rows are distributed among buckets
        total_rows = sum(len(bucket) for bucket in bucketed.values())
        self.assertEqual(total_rows, len(self.data))


if __name__ == "__main__":
    unittest.main()
