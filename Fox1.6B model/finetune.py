# qlora_fox1_1p6b_min_changes.py
# pip install transformers datasets peft bitsandbytes accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- 4-bit quantization config ----------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16   # use torch.bfloat16 if your GPU supports BF16
)

# ---------------- Model and tokenizer ----------------
model_name = "tensoropera/Fox-1-1.6B"  # Fox-1 1.6B base model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# If you are using a single GPU and hit a 4-bit placement error, pin to a device:
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # pick your GPU
# torch.cuda.set_device(0)
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map={"": 0})

# Otherwise, simple load is fine:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_config
)

# Set pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to: {tokenizer.pad_token}")

# Prepare for k-bit training (recommended for QLoRA; does not change your data)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ---------------- LoRA config (Fox/LLaMA-style targets) ----------------
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # optionally add: "gate_proj","up_proj","down_proj"
)
model = get_peft_model(model, lora_config)

# ---------------- Data loading (unchanged) ----------------
dataset = load_dataset(
    "json",
    data_files={
        "train": "/home/hasan/LLM_finetune_military_mutli_drone_planner/spaced_train_Pro_Astar.jsonl",
        "validation": "/home/hasan/LLM_finetune_military_mutli_drone_planner/spaced_valid_Pro_1Astar.jsonl",
    },
)

def preprocess_function(example):
    # Concatenate prompt and response for input (same as your script)
    full_text = example["prompt"] + "\n" + example["response"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    # For causal LM, labels are the same as input_ids (same as your script)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=False)

# ---------------- Training arguments (same structure) ----------------
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    num_train_epochs=5,
    learning_rate=2e-5,
    output_dir="./finetuned_Fox1_1p6B_spaced_Pro_Astar",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    report_to="none",
    fp16=True,                   # switch to bf16=True if supported
    dataloader_pin_memory=True,
    dataloader_num_workers=2
)

# ---------------- Trainer setup (unchanged) ----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

# ---------------- Start fine-tuning ----------------
trainer.train()
