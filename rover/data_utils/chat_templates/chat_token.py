import abc
import random
import typing as ty

import transformers as tr
import datasets as ds


class BaseChatTemplate(abc.ABC):
    def __init__(self, tokenizer: tr.PreTrainedTokenizer, add_bos):
        self.generated_roles = ["assistant"]
        self.tokenizer = tokenizer

        if add_bos:
            self.input_ids = [tokenizer.bos_token_id]
            self.labels = [-100]
            self.str_msg = tokenizer.bos_token
        else:
            self.input_ids = []
            self.labels = []
            self.str_msg = ""

        self.supported_roles = ["user", "assistant", "system"]
        # self.position_ids = []

    def add_message(self, role: str, content: str):
        if role not in self.supported_roles:
            raise NotImplementedError(
                f"role {role} is not implemented for Llama31Tokenizer"
            )
        msg = self.format_msg(content, role)
        self.str_msg += msg
        result = self.tokenizer(msg, add_special_tokens=False)
        if role in self.generated_roles:
            self.labels.extend(result["input_ids"])
        else:
            self.labels.extend([-100] * len(result["input_ids"]))

        self.input_ids.extend(result["input_ids"])

    def add_messages(self, msgs: list):
        for msg in msgs:
            self.add_message(msg["role"], msg["content"])

    @abc.abstractmethod
    def format_msg(self, content, role):
        pass

    @abc.abstractmethod
    def default_system_content(self):
        pass


def validate(tpl: BaseChatTemplate, chat: list = None, output=None):
    if not chat:
        chat = [
            dict(role="user", content="Hi"),
            dict(role="assistant", content="hello"),
        ]

    if not output:
        output = tokenize_chat(tpl, chat, return_text=True)

    expected_str = tpl.tokenizer.apply_chat_template(chat, tokenize=False)
    if expected_str != output["text"]:
        deltas = list(zip(expected_str, output["text"]))
        raise Exception(f"{deltas}")

    expected = tpl.tokenizer.apply_chat_template(
        chat, tokenize=True, add_system_prompt=True
    )
    deltas = list(zip(expected, output["input_ids"]))
    for i, (a, b) in enumerate(deltas):
        if a != b:
            raise Exception(
                f"difference at {i}\nexpected: {(a, b)} everything {deltas}"
            )


def tokenize_chat(
    tpl: BaseChatTemplate,
    chat: ty.List[ty.Dict],
    return_text=True,
    validate_percentage=0.05,
):

    if chat[0]["role"] != "system":
        content = tpl.default_system_content()
        if not content:
            raise ValueError(
                f"add_system_prompt was set but chat template did not define that {tpl}"
            )
        tpl.add_message(
            role="system",
            content=content,
        )

    tpl.add_messages(chat)
    out = dict(
        input_ids=tpl.input_ids,
        label=tpl.labels,
    )
    if return_text:
        out["text"] = tpl.str_msg

    if validate_percentage > 0 and random.random() <= validate_percentage:
        validate(tpl, chat, output=out)
    return out


class CTFactory:
    def __init__(self, klass, tokenizer: tr.PreTrainedTokenizer):
        self.klass = klass
        self.tokenizer = tokenizer

    def new_template(self) -> BaseChatTemplate:
        return self.klass(self.tokenizer)

    def chats_format(self, rec, column_name: str = None):
        if column_name:
            rec = rec[column_name]
        return tokenize_chat(self.new_template(), rec)

    def pairs_format(
        self,
        rec,
        prompt_column: str = "prompt",
        completion_column: str = "completion",
        system_prompt: str = None,
    ):
        msg = []
        if system_prompt:
            msg.append(
                dict(role="user", content=system_prompt),
            )

        msg.append(dict(role="user", content=rec[prompt_column]))
        msg.append(dict(role="assistant", content=rec[completion_column]))

        return self.chats_format(msg)
