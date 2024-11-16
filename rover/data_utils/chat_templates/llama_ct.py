import transformers as tr

from rover.data_utils.chat_templates.chat_token import BaseChatTemplate


class LLama32CT(BaseChatTemplate):
    def __init__(self, tokenizer: tr.PreTrainedTokenizer):
        super().__init__(tokenizer, add_bos=True)

    def format_msg(self, content, role):
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    def default_system_content(self):
        return "Cutting Knowledge Date: December 2023\nToday Date: 04 Nov 2024\n\n"
