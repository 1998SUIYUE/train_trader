import json
from pathlib import Path

log_file = "results/training_history.jsonl"
if Path(log_file).exists():
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    if lines:
        latest = json.loads(lines[-1].strip())
        print("Current Generation:", latest.get('generation', 0))
        print("Best Fitness:", latest.get('best_fitness', 0))
        print("Mean Fitness:", latest.get('mean_fitness', 0))
        print("Generation Time:", latest.get('generation_time', 0), "seconds")
        print("Total Generations:", len(lines))
    else:
        print("No training data found")
else:
    print("Training log file not found")