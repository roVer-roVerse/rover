import unittest

from rover.data_utils.chat_templates.chat_token import validate
from rover.data_utils.chat_templates.llama_ct import LLama32CT
from rover.data_utils.chat_templates.smol_ct import SmolLm2CT
import transformers as tr


class TestSmolLM2(unittest.TestCase):
    def setUp(self):
        pass

    def test_smol(self):
        name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        klass = SmolLm2CT

        t = klass(tr.AutoTokenizer.from_pretrained(name))
        validate(t)

    def test_llama(self):
        name = "meta-llama/Llama-3.2-1B-Instruct"
        klass = LLama32CT

        t = klass(tr.AutoTokenizer.from_pretrained(name))
        validate(t)
