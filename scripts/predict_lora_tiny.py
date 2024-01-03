from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
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
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # quantization_config=quant_config,
        device_map={"": 0},
    )

    model = PeftModel.from_pretrained(base_model, refined_model).merge_and_unload()
    model.eval()

    text_gen = pipeline(
        task="text-generation",
        model=base_model,
        tokenizer=llama_tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )
    # text_gen.model = model

    question = "What is the data model of Redis?"
    context = (
        "Redis is an in-memory database that persists on disk. The data model is key-value, but many different "
        "kind of values are supported: Strings, Lists, Sets, Sorted Sets, Hashes, Streams, HyperLogLogs, "
        "Bitmaps. "
    )

    prompt = prepare_prompt(llama_tokenizer, question=question, context=context)

    print(prompt)

    outputs = text_gen(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False,
    )

    print(outputs[0]["generated_text"])
