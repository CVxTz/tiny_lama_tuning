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

    model = AutoPeftModelForCausalLM.from_pretrained(refined_model, device_map={"": 0}).merge_and_unload()
    model.eval()

    text_gen = pipeline(
        task="text-generation",
        model=base_model,
        tokenizer=llama_tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )
    text_gen.model = model

    _question = "When did Beyonce start becoming popular?"
    _context = """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""

    prompt = prepare_prompt(llama_tokenizer, question=_question, context=_context)

    print(prompt)

    outputs = text_gen(
        prompt.strip(),
        max_new_tokens=256,
        do_sample=True,
        return_full_text=False,
    )

    print(outputs[0]["generated_text"])
