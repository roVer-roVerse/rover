import rover.data_utils.chat_templates.chat_token as ct
from rover.data_utils.loader import load_data
import datasets as ds


def everyday_conversations(tpl: ct.CTFactory, only_keep_output=False, num_proc=None):
    data = load_data("HuggingFaceTB/everyday-conversations-llama3.1-2k")
    data = ds.DatasetDict(
        {
            "train": data["train_sft"],
            "validation": data["test_sft"],
        }
    )

    if tpl:
        data = data.map(
            lambda rec: tpl.chats_format(rec, "messages"),
            num_proc=None,
            remove_columns=data["train"].column_names if only_keep_output else None,
            desc="tokenize: everyday-conversations",
        )
    return data
