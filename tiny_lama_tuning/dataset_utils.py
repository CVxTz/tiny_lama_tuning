import pandas as pd
import torch
from string import Template
from transformers import AutoTokenizer, PreTrainedTokenizer, DefaultDataCollator
from typing import Dict, Any, List


class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        prefix_template: Template,
        tokenizer: PreTrainedTokenizer,
        context_col="context",
        answer_col="answer",
        ignore_index=-100,
        max_prefix_len=1024,
        max_answer_len=32,
    ):
        self.data = data
        self.prefix_template = prefix_template
        self.tokenizer = tokenizer
        self.context_col = context_col
        self.answer_col = answer_col
        self.ignore_index = ignore_index
        self.max_prefix_len = max_prefix_len
        self.max_answer_len = max_answer_len

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        prefix = self.prefix_template.substitute(
            context=self.data.loc[idx, self.context_col]
        )
        answer = self.data.loc[idx, self.answer_col]

        prefix_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(
            prefix, add_special_tokens=False, truncation=True
        )
        prefix_ids = prefix_ids[: self.max_prefix_len]

        answer_ids = self.tokenizer.encode(
            answer, add_special_tokens=False, truncation=True
        ) + [self.tokenizer.eos_token_id]
        answer_ids = answer_ids[: self.max_answer_len]

        labels = [self.ignore_index] * len(prefix_ids) + answer_ids

        return {"input_ids": prefix_ids + answer_ids, "labels": labels}


class CustomDataCollator(DefaultDataCollator):
    def __init__(self, pad_token, ignore_index: int = -100):
        self.pad_token = pad_token
        self.ignore_index = ignore_index

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        batch_input_ids = []
        batch_labels = []

        max_len = max(len(sample_features["input_ids"]) for sample_features in features)

        for sample_features in features:
            input_ids = sample_features["input_ids"]
            labels = sample_features["labels"]

            input_ids += [self.pad_token] * (max_len - len(input_ids))
            labels += [self.ignore_index] * (max_len - len(labels))

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.int64),
            "labels": torch.tensor(batch_labels, dtype=torch.int64),
        }


if __name__ == "__main__":
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )

    print(f"{llama_tokenizer.eos_token_id=}")
    print(f"{llama_tokenizer.pad_token_id=}")

    print(llama_tokenizer.vocab["▁."])
    print(llama_tokenizer._convert_id_to_token(869))

    print(llama_tokenizer.encode("."))
    print(llama_tokenizer.encode_plus(" ."))
    print(llama_tokenizer(" .", return_tensors="pt")["input_ids"].dtype)
    # torch.int64

    template = Template(
        "What is the political bias of this new article?\nContext: $context\nAnswer: "
    )

    df = pd.DataFrame(
        [
            {
                "context": "Obama associate tells liberals to sabotage Trump's election chances at the ballot box",
                "answer": "right",
            }
        ]
    )

    dataset = SFTDataset(data=df, prefix_template=template, tokenizer=llama_tokenizer)

    print(dataset[0])
