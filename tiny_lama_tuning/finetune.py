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

    data_df = (
        pd.DataFrame(data).drop_duplicates(subset=["title", "context"])
    )

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

    print(data_df["text"].values[0])

    data_small = datasets.Dataset.from_pandas(data_df)

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
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    training_data = build_data(llama_tokenizer, split="train")

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
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    # Training Params
    train_params = TrainingArguments(
        output_dir=str(BASE_PATH / "results_modified"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        save_steps=1000,
        logging_steps=25,
        learning_rate=1e-5,
        fp16=True,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=False,
        lr_scheduler_type="constant",
    )
    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params,
    )
    # Training
    fine_tuning.train()
    # Save Model
    fine_tuning.model.save_pretrained(refined_model)
