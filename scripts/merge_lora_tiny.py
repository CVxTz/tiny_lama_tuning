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


adapter_model_name = "../tiny_lama_tuning/TinyLlama-1.1B-refined"
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
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    # quantization_config=quant_config,
    device_map={"": 0},
)
base_model.config.pretraining_tp = 1

# Load the PEFT model
model = PeftModel.from_pretrained(base_model, adapter_model_name)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(output_name)

query = "How do I use the OpenAI API?"
text_gen = pipeline(
    task="text-generation", model=model, tokenizer=llama_tokenizer, max_length=200
)
output = text_gen(f"<s>[INST] {query} [/INST]")
print(output[0]["generated_text"])
