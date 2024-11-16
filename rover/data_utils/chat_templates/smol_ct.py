from textwrap import dedent

import transformers as tr

from rover.data_utils.chat_templates.chat_token import BaseChatTemplate


class SmolLm2CT(BaseChatTemplate):
    def __init__(self, tokenizer: tr.PreTrainedTokenizer):
        super().__init__(tokenizer, add_bos=False)

    def format_msg(self, content, role):
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def default_system_content(self):
        return "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
