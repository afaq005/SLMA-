# qlora_gemma2b_cuda1_min_changes.py
# pip install transformers datasets peft bitsandbytes accelerate

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ensure this process sees only GPU:1

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization config (same as yours)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # use torch.bfloat16 if your GPU supports BF16
)

# Model and tokenizer
model_name = "google/gemma-2-2b"  # or "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Pin model to CUDA:1 explicitly; avoid device_map="auto" for single-GPU 4-bit
assert torch.cuda.is_available()
torch.cuda.set_device(0)  # with CUDA_VISIBLE_DEVICES=1, index 0 maps to physical GPU:1
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": 0},  # index 0 inside this process == physical GPU:1
)

# Set pad_token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to: {tokenizer.pad_token}")

# Prepare for k-bit training (minimal change for stability)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# LoRA config (keep as in your script)
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# Load data (unchanged)
dataset = load_dataset(
    "json",
    data_files={
        "train": "/home/hasan/LLM_finetune_military_mutli_drone_planner/spaced_train_Pro_Astar.jsonl",
        "validation": "/home/hasan/LLM_finetune_military_mutli_drone_planner/spaced_valid_Pro_1Astar.jsonl",
    },
)

def preprocess_function(example):
    full_text = example["prompt"] + "\n" + example["response"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=False)

# TrainingArguments aligned to GPU in this process (no other changes)
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    num_train_epochs=5,
    learning_rate=2e-5,
    output_dir="./finetuned_gemma2b_newthis_spaced_Pro_Astar",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    report_to="none",
    fp16=True,                 # switch to bf16=True if supported
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()
