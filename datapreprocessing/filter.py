# validate_reach_and_avoid_jsonl_nb.py
import json, re
from collections import defaultdict

def parse_pr_record(rec):
    """
    Input: {"prompt": "...", "response": "..."}
    Returns: (objective tuple, set(threat tuples), dict name->list[(x,y)])
    """
    prompt = rec.get("prompt", "")
    response = rec.get("response", "")

    # Objective (x,y) [type]
    m_obj = re.search(r"Objective at \((\d+),\s*(\d+)\)\s*\[[^\]]+\]", prompt)
    if not m_obj:
        raise ValueError("objective_not_found")
    obj = (int(m_obj.group(1)), int(m_obj.group(2)))

    # Threats: (x,y) [type], ...
    threats = set()
    m_thr = re.search(r"Threats at (.+?)\.\s*Objective", prompt)
    if m_thr:
        thr_str = m_thr.group(1)
        for x, y in re.findall(r"\((\d+),\s*(\d+)\)\s*\[[^\]]+\]", thr_str):
            threats.add((int(x), int(y)))

    # Paths per drone line
    waypoints = {}
    for line in response.strip().splitlines():
        m_name = re.match(r"(Drone\d+):", line.strip())
        if not m_name:
            continue
        name = m_name.group(1)
        coords = [(int(x), int(y)) for x, y in re.findall(r"\((\d+),\s*(\d+)\)", line)]
        waypoints[name] = coords

    # Normalize order Drone1, Drone2, ...
    waypoints = {k: waypoints[k] for k in sorted(waypoints.keys(),
                                                 key=lambda z: int(re.findall(r"\d+", z)[0]))}
    return obj, threats, waypoints

def is_valid_reach_and_avoid(obj, threats, waypoints):
    """
    Only two checks:
      1) Each drone path ends exactly at the objective.
      2) No waypoint equals any threat cell.
    """
    for name, path in waypoints.items():
        if not path:
            return False, f"{name}: empty_path"
        if tuple(path[-1]) != obj:
            return False, f"{name}: not_ending_at_objective"
        for p in path:
            if p in threats:
                return False, f"{name}: steps_on_threat_{p}"
    return True, "ok"

def filter_jsonl_reach_avoid(in_path, out_path):
    """
    Reads JSONL prompt/response records, keeps only those passing reach-and-avoid,
    writes a JSONL with the same records, and returns a summary dict.
    """
    total = 0
    valid = 0
    errors = defaultdict(int)
    clean = []

    with open(in_path, "r") as fin:
        for ln, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
                obj, threats, waypoints = parse_pr_record(rec)
                ok, msg = is_valid_reach_and_avoid(obj, threats, waypoints)
                if ok:
                    clean.append(rec)
                    valid += 1
                else:
                    errors[msg] += 1
            except Exception as e:
                errors[f"parse_error:{str(e)}"] += 1

    with open(out_path, "w") as fout:
        for r in clean:
            fout.write(json.dumps(r) + "\n")

    summary = {
        "total": total,
        "valid": valid,
        "dropped": total - valid,
        "top_issues": sorted(errors.items(), key=lambda x: -x[1])[:10],
        "outfile": out_path
    }
    print(f"Total: {summary['total']} | Valid: {summary['valid']} | Dropped: {summary['dropped']}")
    if summary["top_issues"]:
        print("Top issues:")
        for k, v in summary["top_issues"]:
            print(f"  {v}x - {k}")
    print(f"Clean JSONL written to: {out_path}")
    return summary
