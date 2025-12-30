import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model




# Quantization config for 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Model and tokenizer
# model_name = "microsoft/phi-2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",  # Automatically uses your GPU
#     quantization_config=quant_config
# )


model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_config
)


# Set pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to: {tokenizer.pad_token}")

# LoRA config
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# Load your data
dataset = load_dataset('json', data_files={'train':"/home/hasan/LLM_finetune_military_mutli_drone_planner/spaced_train_Pro_Astar.jsonl", 'validation': "/home/hasan/LLM_finetune_military_mutli_drone_planner/spaced_valid_Pro_1Astar.jsonl"})


def preprocess_function(example):
    # Concatenate prompt and response for input
    full_text = example["prompt"] + "\n" + example["response"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=False)


# Training arguments (no 'device' argument needed)
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    num_train_epochs=5,
    learning_rate=2e-5,
    output_dir="./finetuned_tinyllama_spaced_Pro_Astar",
    eval_strategy="epoch",  # or 'eval_strategy' for older transformers
    save_strategy="epoch",
    logging_steps=1,
    report_to="none",
    fp16=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=2
)

# Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset['train'],
#     eval_dataset=dataset['validation'],
#     tokenizer=tokenizer
# )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer
)



# Start fine-tuning
trainer.train()
