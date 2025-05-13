import os 
import json


for ix in [1]:
    for e in ["codecontests", "connections", "sequences"]:
        with open(f"finished_experiments/{e}1_BASE/prompts.jsonl", "r") as f:
            prompts = [json.loads(line) for line in f.readlines()]
            prompts = sorted(prompts, key=lambda x: x["test_score"], reverse=True)
        initial = prompts[10:20]
        #for op in ["REFLECTIVE", "ITERATIVE", "FEEDBACK", "PARAPHRASE"]:
        for op in ["REFLECTIVE"]:
            os.makedirs(f"runs/{e}{ix}_{op}")
            with open(f"runs/{e}{ix}_{op}/prompts.jsonl", "w") as f:
                for p in initial:
                    json.dump(p, f)
                    f.write("\n")
