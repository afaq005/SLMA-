
import numpy as np
import heapq
import random
import json

GRID_SIZE = 40   

def random_coords_2d(num, exclude=[]):
    coords = []
    while len(coords) < num:
        c = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if c not in coords and c not in exclude:
            coords.append(c)
    return coords

def heuristic(a, b):
    # Allow ((x, y), t) or (x, y)
    if isinstance(a, (tuple, list)) and len(a) == 2 and isinstance(a[0], (tuple, list)):
        a = a[0]
    if isinstance(b, (tuple, list)) and len(b) == 2 and isinstance(b[0], (tuple, list)):
        b = b[0]
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_2d_dynamic(grid, start, goal, dynamic_obstacles):
    open_set = []
    heapq.heappush(open_set, (0, 0, start))
    came_from = {}
    g_score = { (start,0): 0 }
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    visited = set()
    max_steps = GRID_SIZE*4

    while open_set:
        _, t, curr = heapq.heappop(open_set)
        if curr == goal:
            return reconstruct_path_time(came_from, curr, t)
        if (curr, t) in visited:
            continue
        visited.add((curr, t))
        for d in directions:
            neighbor = (curr[0]+d[0], curr[1]+d[1])
            t_next = t + 1
            if (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                if grid[neighbor] != 0:
                    continue
                if (neighbor, t_next) in dynamic_obstacles:
                    continue
                key = (neighbor, t_next)
                tentative_g = g_score.get((curr, t), np.inf) + 1
                if tentative_g < g_score.get(key, np.inf):
                    came_from[key] = ((curr, t))
                    g_score[key] = tentative_g
                    heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), t_next, neighbor))
            if t_next > max_steps:
                break
    return []

def reconstruct_path_time(came_from, curr, t):
    path = [(curr, t)]
    key = (curr, t)
    while key in came_from:
        key = came_from[key]
        path.append(key)
    path = list(reversed(path))
    return [pt for pt, _ in path]

def make_grid_2d(obstacles):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for obs in obstacles:
        grid[obs] = 1
    return grid

def scenario_sample_2d():
    positions = random_coords_2d(3)
    objective = random_coords_2d(1, exclude=positions)[0]
    threats = random_coords_2d(4, exclude=positions + [objective])
    grid = make_grid_2d(threats)
    waypoints = {}
    dynamic_obstacles = set()
    for i, drone in enumerate(positions):
        path = a_star_2d_dynamic(grid, drone, objective, dynamic_obstacles)
        if not path or len(path) < 2:
            return None  # Unsolvable configuration
        for t, pt in enumerate(path):
            dynamic_obstacles.add((pt, t))
        waypoints[f'drone{i+1}'] = path
    return {
        "input": {
            "drone_positions": positions,
            "objective": objective,
            "threats": threats
        },
        "output": {
            "waypoints": waypoints
        }
    }

def generate_dataset_2d(num_samples=3000):
    data = []
    tries = 0
    max_tries = 20 * num_samples  # More attempts for sparser valid configs
    while len(data) < num_samples and tries < max_tries:
        sample = scenario_sample_2d()
        if sample:
            data.append(sample)
            print(f"Generated {len(data)}/{num_samples}")
        tries += 1
    with open('dataset_2d_Pro_A*_3000.json', 'w') as f:
        json.dump(data, f)
    print(f"Finished. Successfully generated {len(data)} samples.")

if __name__ == "__main__":
    generate_dataset_2d(3000)   
