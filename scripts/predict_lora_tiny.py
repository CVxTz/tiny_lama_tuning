from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
LlamaForCausalLM
)
from time import perf_counter
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def catch_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.3f} seconds")


def prepare_prompt(tokenizer, question, context):
    messages = [
        {
            "role": "user",
            "content": f"Question: {question}\nContext: {context}",
        }
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


if __name__ == "__main__":
    BASE_PATH = Path(__file__).parents[1] / "outputs"

    refined_model = str(BASE_PATH / "TinyLlama-1.1B-refined")
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    # llama_tokenizer.pad_token = llama_tokenizer.eos_token
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # quantization_config=quant_config,
        device_map={"": 0},
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        refined_model, device_map={"": 0}
    ).merge_and_unload()
    model.eval()

    text_gen = pipeline(
        task="text-generation",
        model=base_model,
        tokenizer=llama_tokenizer,
    )
    text_gen.model = model

    _context = "Obama associate tells liberals to sabotage Trump's election chances at the ballot box"
    prompt = f"What is the political bias of this new article?\nContext: {_context}\nAnswer: "

    print(prompt)

    outputs = text_gen(
        prompt.strip(),
        do_sample=False,
        return_full_text=False,
        max_new_tokens=20
    )

    print(outputs[0]["generated_text"])
