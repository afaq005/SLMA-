import json
from sklearn.model_selection import train_test_split

# Load your 30-sample dataset
with open("/home/hasan/LLM_finetune_military_mutli_drone_planner/multi_drone_four_threats_spaced_ft_enriched.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]

# 90/10 split
train, valid = train_test_split(data, test_size=0.3, random_state=42)

# Save splits
with open('spaced_train_Pro_Astar.jsonl', 'w') as f:
    for item in train:
        f.write(json.dumps(item) + '\n')
with open('spaced_valid_Pro_Asta.jsonl', 'w') as f:
    for item in valid:
        f.write(json.dumps(item) + '\n')

print(f"Train size: {len(train)}, Valid size: {len(valid)}")
