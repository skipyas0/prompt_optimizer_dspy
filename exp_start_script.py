import os 
import json

ix = 1
for e in ["codecontests", "connections", "sequences"]:
    with open(f"runs/{e}{ix}/prompts.jsonl", "r") as f:
        prompts = [json.loads(line) for line in f.readlines()]
        prompts = sorted(prompts, key=lambda x: x["test_score"], reverse=True)
    initial = prompts[10:20]
    for op in ["REFLECTIVE", "ITERATIVE", "FEEDBACK", "PARAPHRASE"]:
        os.makedirs(f"runs/{e}{ix}_{op}")
        with open(f"runs/{e}{ix}_{op}/prompts.jsonl", "w") as f:
            for p in initial:
                json.dump(p, f)
                f.write("\n")
