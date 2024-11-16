import unittest
from unittest import TestCase

import transformers as tr


try:
    from rover.models.te_llama.utils import load_te_from_hf
    from rover.models.te_llama.te_llama import save_model

    _te_installed = True
except ImportError:
    _te_installed = False


@unittest.skipIf(not _te_installed, "TE not available")
class Test(TestCase):
    def test_init_te_llama_model(self):
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        te_model = load_te_from_hf(model_name)

        self.verify_basics(model_name, te_model)

        save_path = "/tmp/anuj1"
        save_model(te_model, save_path)

        del te_model

        hf_model = tr.AutoModelForCausalLM.from_pretrained(save_path)
        self.verify_basics(model_name, hf_model)

    def verify_basics(self, model_name, model):
        pipe = tr.pipeline(
            "text-generation",
            model=model,
            tokenizer=tr.AutoTokenizer.from_pretrained(model_name),
            device="cuda",
        )
        model.use_cache = False
        output = pipe(
            [dict(role="user", content="what is 2 + 3")], max_length=60, use_cache=False
        )
        self.assertIn(output[0]["generated_text"][1]["content"], ["2 + 3 = 5"])
