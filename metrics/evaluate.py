import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import math
import evaluate
import re
import time

# 1. Load Model and Tokenizer
model_dir = "/home/hasan/LLM_finetune_military_mutli_drone_planner/finetuned_tinyllama_spaced_Pro_Astar/checkpoint-10500/"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda:3",
    quantization_config=quant_config
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load and Prepare Eval Dataset
eval_dataset = load_dataset('json', data_files={'eval': 'spaced_test_Pro_Asta.jsonl'})['eval']

# 3. Compute Model Loss (for Perplexity)
def preprocess_function(example):
    full_text = example["prompt"] + "\n" + example["response"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=300,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_eval = eval_dataset.map(preprocess_function)

with torch.no_grad():
    total_loss = 0
    count = 0
    for batch in tokenized_eval:
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(model.device)
        labels = torch.tensor(batch["labels"]).unsqueeze(0).to(model.device)
        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item()
        count += 1
    avg_loss = total_loss / count if count > 0 else float('nan')
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

print(f"Eval loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")

# 4. Compute BLEU, ROUGE, BERTScore & Token-per-Second Throughput
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')

predictions = []
references = []

total_gen_tokens = 0
total_gen_time = 0.0

for example in eval_dataset:
    prompt = example["prompt"]
    reference = example["response"]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Timing start
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=1400, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - start_time
    # End timing

    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append(pred)
    references.append([reference])  # BLEU expects a list of references per prediction

    num_new_tokens = output.shape[-1] - input_ids.shape[-1]
    total_gen_tokens += num_new_tokens
    total_gen_time += elapsed

tokens_per_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0
print(f"Avg tokens/sec: {tokens_per_sec:.2f} (Generated {total_gen_tokens} tokens in {total_gen_time:.2f} seconds)")

bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=[ref[0] for ref in references])
bertscore_result = bertscore.compute(predictions=predictions, references=[ref[0] for ref in references], lang="en")

print(f"BLEU score: {bleu_score['bleu']:.4f}")
print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
print(f"BERTScore (F1): {sum(bertscore_result['f1'])/len(bertscore_result['f1']):.4f}")

# 5. Task-Specific Metrics (as before)

def extract_coordinates(text):
    return re.findall(r"\((\d+),\s*(\d+)\)", text)

def extract_drone_paths(response):
    drones = {}
    lines = response.split('\n')
    current_drone = None
    for line in lines:
        match = re.match(r"Drone\s*([0-9]+)", line, re.IGNORECASE)
        if match:
            current_drone = match.group(1)
            drones[current_drone] = []
        elif current_drone is not None:
            coords = extract_coordinates(line)
            drones[current_drone].extend(coords)
    return drones

def check_success(prediction, prompt):
    num_drones_match = re.search(r"(\d+) drones", prompt)
    if not num_drones_match:
        return 0
    num_drones = int(num_drones_match.group(1))
    drone_ids = [str(i+1) for i in range(num_drones)]
    drone_paths = extract_drone_paths(prediction)
    for did in drone_ids:
        if did not in drone_paths or len(drone_paths[did]) == 0:
            return 0
    return 1

def check_threat_avoidance(prediction, prompt):
    threat_coords = []
    if "Threats at" in prompt:
        threat_coords = extract_coordinates(prompt.split("Threats at")[1])
    drone_paths = extract_drone_paths(prediction)
    for path in drone_paths.values():
        for coord in path:
            if coord in threat_coords:
                return 0
    return 1

def check_coverage(prediction, prompt):
    num_drones_match = re.search(r"(\d+) drones", prompt)
    if not num_drones_match:
        return 0
    num_drones = int(num_drones_match.group(1))
    drone_ids = [str(i+1) for i in range(num_drones)]
    drone_paths = extract_drone_paths(prediction)
    covered = sum(1 for did in drone_ids if did in drone_paths and len(drone_paths[did]) > 0)
    return covered / num_drones if num_drones > 0 else 0

def ngram_diversity(preds, n=4):
    all_ngrams = set()
    for pred in preds:
        tokens = pred.split()
        ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        all_ngrams.update(ngrams)
    return len(all_ngrams) / max(1, len(preds))

def avg_length(preds):
    return sum(len(pred.split()) for pred in preds) / len(preds) if preds else 0

# Aggregate metrics
successes = []
threat_avoids = []
coverages = []

for pred, example in zip(predictions, eval_dataset):
    prompt = example["prompt"]
    successes.append(check_success(pred, prompt))
    threat_avoids.append(check_threat_avoidance(pred, prompt))
    coverages.append(check_coverage(pred, prompt))

success_rate = sum(successes) / len(successes) if successes else 0
threat_avoid_rate = sum(threat_avoids) / len(threat_avoids) if threat_avoids else 0
avg_coverage = sum(coverages) / len(coverages) if coverages else 0
diversity_4gram = ngram_diversity(predictions, n=4)
avg_out_len = avg_length(predictions)

print(f"Success Rate: {success_rate:.2%}")
print(f"Threat Avoidance Rate: {threat_avoid_rate:.2%}")
print(f"Average Drone Coverage: {avg_coverage:.2%}")
print(f"4-gram Diversity: {diversity_4gram:.4f}")
print(f"Average Output Length: {avg_out_len:.1f} words")

# Extra: Device and environment info
print(f"Model device: {model.device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Max memory allocated (MB): {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f}")
