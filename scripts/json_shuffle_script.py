import json
import random
fp = "datasets/connections.json"
out = "datasets/connections1.json"
with open(fp, "r") as f:
    data = json.load(f)

random.shuffle(data)
print(data)
with open(out, "w") as f:
    json.dump(data, f, indent=4)