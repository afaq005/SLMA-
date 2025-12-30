# convert_and_enrich_prioritized_astar_to_ft_jsonl.py
import json
import random
import sys
import argparse

# Vocabularies (exactly as requested)
THREAT_TYPES = [
    "static SAM site", "moving vehicle", "enemy drone", "radar jammer",
    "anti-air gun", "hostile patrol", "GPS jammer"
]

OBJECTIVE_TYPES = [
    "rescue", "delivery", "survey", "decoy", "recon", "track", "defense"
]

TERRAIN_TYPES = [
    "urban", "forest", "desert", "mountain", "coastal", "rural", "industrial"
]

WEATHER_TYPES = [
    "clear", "rainy", "foggy", "windy", "stormy", "night", "dawn", "dusk"
]

def enrich_sample(sample, rng):
    """
    Returns a new dict with added:
      input.terrain, input.weather,
      input.objective_type,
      input.threat_types (parallel to input.threats).
    Does not modify the original sample in-place.
    """
    s = json.loads(json.dumps(sample))  # deep copy (safe for primitives)
    inp = s["input"]

    # Random assignments
    inp["terrain"] = rng.choice(TERRAIN_TYPES)
    inp["weather"] = rng.choice(WEATHER_TYPES)
    inp["objective_type"] = rng.choice(OBJECTIVE_TYPES)

    # Assign a type to each threat position
    threat_types = [rng.choice(THREAT_TYPES) for _ in inp["threats"]]
    inp["threat_types"] = threat_types

    return s

def sample_to_prompt_response(enriched_sample):
    """
    Build the FT prompt/response string pair using enriched metadata.
    """
    inp = enriched_sample["input"]
    starts = inp["drone_positions"]
    threats = inp["threats"]
    threat_types = inp.get("threat_types", ["threat"] * len(threats))
    obj = inp["objective"]
    obj_type = inp.get("objective_type", "rescue")
    terrain = inp.get("terrain", "urban")
    weather = inp.get("weather", "clear")

    starts_txt = ", ".join([f"({x},{y})" for x, y in starts])
    threats_txt = ", ".join([f"({x},{y}) [{t}]" for (x, y), t in zip(threats, threat_types)])

    prompt = (
        f"{len(starts)} drones at {starts_txt}. "
        f"Threats at {threats_txt}. "
        f"Objective at ({obj[0]},{obj[1]}) [{obj_type}]. "
        f"Terrain: {terrain}. Weather: {weather}."
    )

    waypoints = enriched_sample["output"]["waypoints"]
    lines = []
    # Ensure a stable order: Drone1, Drone2, Drone3 (if keys are drone1, drone2, ...)
    for i, name in enumerate(sorted(waypoints.keys())):
        path = waypoints[name]
        trail = " -> ".join([f"({x},{y})" for x, y in path])
        lines.append(f"Drone{i+1}: {trail} [{obj_type}];")
    response = "\n".join(lines)
    return {"prompt": prompt, "response": response}

def convert_dataset(
    infile="multi_drone_planner_dataset_2d1.json",
    outfile_jsonl="multi_drone_prioritized_astar_ft_enriched.jsonl",
    outfile_enriched_json="multi_drone_prioritized_astar_enriched.json",
    seed=None
):
    rng = random.Random(seed)

    with open(infile, "r") as f:
        data = json.load(f)

    enriched = []
    count = 0
    with open(outfile_jsonl, "w") as fjsonl:
        for s in data:
            es = enrich_sample(s, rng)
            enriched.append(es)
            pr = sample_to_prompt_response(es)
            fjsonl.write(json.dumps(pr) + "\n")
            count += 1

    with open(outfile_enriched_json, "w") as fj:
        json.dump(enriched, fj)

    print(f"Wrote {count} prompt/response lines to {outfile_jsonl}")
    print(f"Wrote enriched JSON with metadata to {outfile_enriched_json}")

def main():
    # Make argparse notebook-friendly by stripping Jupyter's injected -f/--f
    if any(a.startswith("-f") or a.startswith("--f=") for a in sys.argv):
        sys.argv = [sys.argv[0]]

    ap = argparse.ArgumentParser(
        description="Enrich Prioritized A* dataset with random threat/objective types, terrain, weather, and export FT JSONL."
    )
    ap.add_argument("--infile", type=str, default="multi_drone_four_threats_spaced.json")
    ap.add_argument("--outfile_jsonl", type=str, default="multi_drone_four_threats_spaced_ft_enriched.jsonl")
    ap.add_argument("--outfile_enriched_json", type=str, default="multi_drone_four_threats_spaced_enriched.json")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    convert_dataset(
        infile=args.infile,
        outfile_jsonl=args.outfile_jsonl,
        outfile_enriched_json=args.outfile_enriched_json,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
