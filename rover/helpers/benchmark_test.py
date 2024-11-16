import unittest
from unittest import TestCase
import numpy as np


class TestBench(TestCase):
    def test_bench(self):
        xs = np.random.random([100, 100, 100])
