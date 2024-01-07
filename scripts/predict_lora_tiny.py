from peft import AutoPeftModelForCausalLM
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from time import perf_counter
from contextlib import contextmanager
from pathlib import Path
from tiny_lama_tuning.dataset_utils import build_data


@contextmanager
def catch_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.3f} seconds")


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
    base_model = LlamaForCausalLM.from_pretrained(
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

    test_data = build_data(llama_tokenizer, split="test")

    sample = test_data.get_sample(10)
    prompt, answer = sample["prompt"], sample["answer"]

    print(f"{prompt=}")
    print(f"{answer=}")

    outputs = text_gen(
        prompt,
        do_sample=False,
        return_full_text=False,
        max_new_tokens=200,
        # logits_processor=get_logit_criteria(tokenizer=llama_tokenizer),
    )

    print(f"""{outputs[0]["generated_text"]=}""")
