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


@contextmanager
def catch_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.3f} seconds")


base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_name = "../tiny_lama_tuning/TinyLlama-1.1B-merged"

llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.padding_side = "right"  # Fix for fp16
# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
# Model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0},
)

query = "How do I use the OpenAI API?"
text_gen = pipeline(
    task="text-generation",
    model=model,
    tokenizer=llama_tokenizer,
    max_new_tokens=256,
    do_sample=False,
)
output = text_gen(f"<s>[INST] {query} [/INST]")

with catch_time():
    output = text_gen(f"<s>[INST] {query} [/INST]")

with catch_time():
    output = text_gen(f"<s>[INST] {query} [/INST]")
    print(output[0]["generated_text"])
