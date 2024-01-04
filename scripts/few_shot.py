import torch
from time import perf_counter
from contextlib import contextmanager
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


@contextmanager
def catch_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.3f} seconds")


if __name__ == "__main__":
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # Tokenizer
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
        base_model_name, device_map={"": 0}, quantization_config=quant_config
    )
    base_model.config.use_cache = True

    pytorch_total_params = base_model.num_parameters()
    print(pytorch_total_params)

    pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=llama_tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    few_shot_messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot that uses the context to answer questions with the minimum amount of"
            " words",
        },
        {
            "role": "user",
            "content": "Question: What year was jimmy born ?\n"
            "Context: Jimmy Donal Wales (born August 7, 1966), also known on Wikipedia by the "
            "nickname Jimbo Wales, is an American and British[3] Internet entrepreneur, "
            "webmaster, and former financial trader.",
        },
        {
            "role": "assistant",
            "content": "1966",
        },
        {
            "role": "user",
            "content": "Question: What does MDP stand for ?\n"
            "Context: In mathematics, a Markov decision process (MDP) is a discrete-time stochastic control "
            "process.",
        },
        {
            "role": "assistant",
            "content": "Markov decision process",
        },
    ]

    messages = few_shot_messages + [
        {
            "role": "user",
            "content": "Question: What is the data model of Redis ?\n"
            "Context: Redis is an in-memory database that persists on disk. The data model is key-value, "
            "but many different kind of values are supported: Strings, Lists, Sets, Sorted Sets, Hashes, "
            "Streams, HyperLogLogs, Bitmaps. ",
        }
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(prompt)
    with catch_time():
        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            return_full_text=False,
        )
        print(outputs[0]["generated_text"])
