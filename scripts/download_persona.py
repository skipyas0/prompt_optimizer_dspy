import datasets
import json

def download_persona():
    ds = datasets.load_dataset("proj-persona/PersonaHub", "reasoning")['train']
    ds = ds.filter(lambda x: x["input persona"][:2] == "A ")
    ds = ds.filter(lambda x: any([w in x["input persona"] for w in ["student", "professor", "teacher", "researcher", "programmer", "scientist", "writer", "artist", "engineer"]]))
    ds = ds.filter(lambda x: x["description"] == "logical reasoning" and len(x["input persona"]) > 130 and len(x["input persona"]) < 200)
    ds = ds.remove_columns(["synthesized text", "description"])
    ds = ds.shuffle(seed=42)[:100]["input persona"]
    print(ds)

    with open("datasets/personas.json", "w", encoding="utf-8") as f:
        json.dump(ds, f, indent = 4)

def download_codecontests():
    ds = datasets.load_dataset('deepmind/code_contests', split='train')
    ds = ds.filter(lambda ex: ex['difficulty']  == 7 and '<image>' not in ex['description'] and 1 in ex["solutions"]["language"]) # filter easy samples 
    def map_code_contests(example):
        question = example['description']
        test_inputs = [x.strip().split('\n') for x in example['private_tests']['input']]
        test_outputs = [x.strip() for x in example['private_tests']['output']]
        python_index = example['solutions']['language'].index(1)
        code = example['solutions']['solution'][python_index]
        return {
            'question': question,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs,
            'code': code
        }
    
    ds = ds.map(map_code_contests, remove_columns=ds.column_names, load_from_cache_file=False)
    ds = ds.select(range(30)).to_list()
    with open("datasets/codecontests.json", "w+") as f:
        json.dump(ds, f)

