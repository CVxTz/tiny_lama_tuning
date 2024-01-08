from peft import AutoPeftModelForCausalLM
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextGenerationPipeline,
)
from tqdm import tqdm
from pathlib import Path
from tiny_lama_tuning.dataset_utils import build_data


class PatchedTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(
        self,
        prompt_text,
        prefix="",
        handle_long_generation=None,
        add_special_tokens=False,
        **generate_kwargs,
    ):
        # Patch issue https://github.com/huggingface/transformers/issues/27869 :/
        inputs = self.tokenizer(
            prefix + prompt_text,
            padding=False,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            truncation=True,
        )
        inputs["prompt_text"] = prompt_text

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = (
                    generate_kwargs.get("max_length", self.model.config.max_length)
                    - cur_len
                )
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][
                        :, -keep_length:
                    ]

        return inputs


if __name__ == "__main__":
    BASE_PATH = Path(__file__).parents[1] / "outputs"

    refined_model = str(BASE_PATH / "TinyLlama-1.1B-refined")
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    llama_tokenizer.truncation_side = "left"

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Model
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map={"": 0},
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        refined_model,
        device_map={"": 0},  # , quantization_config=quant_config
    )
    model.eval()

    text_gen = PatchedTextGenerationPipeline(
        model=base_model,
        tokenizer=llama_tokenizer,
    )
    text_gen.model = model

    test_data = build_data(llama_tokenizer, split="test")

    correct_count = 0

    for i in tqdm(range(len(test_data))):
        sample = test_data.get_sample(i)
        prompt, answer = sample["prompt"], sample["answer"]

        # print(f"{prompt=}")
        # print(f"{answer=}")

        outputs = text_gen(
            prompt,
            do_sample=False,
            return_full_text=False,
            max_new_tokens=200,
        )

        correct = answer == outputs[0]["generated_text"].strip()

        correct_count += correct

        print(
            f"""{answer=} / {outputs[0]["generated_text"]=} / {correct=} / {correct_count / (i+1)}"""
        )

    accuracy = 100 * correct_count / len(test_data)

    print(f"{accuracy=}")
    # accuracy=73.60594795539033
