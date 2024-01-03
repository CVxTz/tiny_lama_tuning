import torch
from transformers import pipeline

if __name__ == "__main__":
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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

    outputs = pipe(
        prompt,
        max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, return_full_text=False
    )
    print(outputs[0]["generated_text"])
