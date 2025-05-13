import uuid
import json
import os

datasets = os.listdir()
for dataset in datasets:
    if dataset.endswith(".json"):
        with open(dataset, "r") as f:
            data = json.load(f)
        
        for example in data:
            example["id"] = uuid.uuid4().hex
        
        with open(dataset, "w") as f:
            json.dump(data, f, indent=4)