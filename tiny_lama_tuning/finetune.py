import datasets
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from pathlib import Path
import pandas as pd

MISSING_ANSWER = "Unanswerable"


def build_data(tokenizer, split="train"):
    data_name = "squad_v2"
    data = load_dataset(data_name, split=split)

    data_df = pd.DataFrame(data).drop_duplicates(subset=["title", "context"])

    data_df["text"] = data_df.apply(
        lambda x: tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"Question: {x['question']}\nContext: {x['context']}",
                },
                {
                    "role": "assistant",
                    "content": x["answers"]["text"][0]
                    if x["answers"]["text"]
                    else MISSING_ANSWER,
                },
            ],
            tokenize=False,
            add_generation_prompt=False,
        ),
        axis=1,
    )

    data_small = data_df.sample(16).reset_index(drop=True)

    print(data_small["text"].values[0])

    data_small = datasets.Dataset.from_pandas(data_small)

    return data_small


if __name__ == "__main__":
    BASE_PATH = Path(__file__).parents[1] / "outputs"

    BASE_PATH.mkdir(exist_ok=True)

    # Model and tokenizer names
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    refined_model = str(BASE_PATH / "TinyLlama-1.1B-refined")
    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )

    training_data = build_data(llama_tokenizer, split="train")

    print(f"{len(training_data)=}")

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": 0},
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head']
    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    # Training Params
    train_params = TrainingArguments(
        output_dir=str(BASE_PATH / "results_modified"),
        num_train_epochs=200,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=1000,
        logging_steps=25,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.05,
        fp16=True,
        max_steps=-1,
        group_by_length=False,
    )
    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params,
        max_seq_length=llama_tokenizer.model_max_length,
    )

    print(fine_tuning.model.print_trainable_parameters())
    # Training
    fine_tuning.train()
    # Save Model
    fine_tuning.model.save_pretrained(refined_model)
