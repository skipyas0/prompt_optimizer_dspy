import builtins
from datasets import load_dataset
import json
import os
import io
from contextlib import redirect_stdout
import traceback
import multiprocessing
import resource
import ast
from typing import Type, TypeVar, Any, get_origin, get_args

SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,

    # math functions
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "pow": pow,

    # basic types
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,

    # sequence operations
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,

    # string operations
    "chr": chr,
    "ord": ord,
    
    # import-safe modules
    "__import__": lambda name, *args: None,  # disable imports
}

def download_gsm8k():
    ds = load_dataset('openai/gsm8k', 'main', split='train').select(range(15))

    def map_gsm8k(example):
        example['question'] = example['question']
        example['answer'] = example['answer'].split('####')[1].replace('\xa0', '').strip()
        return {'question': example['question'], 'answer': example['answer']}
    
    ds = ds.map(map_gsm8k, remove_columns=ds.column_names, load_from_cache_file=False).to_list()
    with open("gsm8k.json", "w+") as f:
        json.dump(ds, f)



def load_gsm8k_server():
    ds = load_dataset('openai/gsm8k', 'main', split='train').select(range(15))

    def map_gsm8k(example):
        example['question'] = example['question']
        example['answer'] = example['answer'].split('####')[1].replace('\xa0', '').strip()
        return {'question': example['question'], 'answer': example['answer']}
    
    ds = ds.map(map_gsm8k, remove_columns=ds.column_names, load_from_cache_file=False).to_list()

    return list(ds)

def download_codecontests():
    ds = load_dataset('deepmind/code_contests', split='train')
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
    with open("codecontests.json", "w+") as f:
        json.dump(ds, f)

def check_and_load_population(folder: str) -> list:
    initial_population = []
    initial_attempts = []
    initial_comparisons = []
    if os.path.exists(folder):
        if os.path.exists(folder+'/prompts.jsonl'):
            from prompt import Prompt   
            with open(folder+'/prompts.jsonl', 'r') as f:
                prompts = [json.loads(line) for line in f.readlines()]
                initial_population = [Prompt.from_json(p) for p in prompts]
        if os.path.exists(folder+'/task_attempts.jsonl'):
            from data import Attempt
            with open(folder+'/task_attempts.json', 'r') as f:
                attempts = [json.loads(line) for line in f.readlines()]
                initial_attempts = [Attempt.from_json(a) for a in attempts]
                for p in initial_population:
                    p.attempts = [a for a in initial_attempts if a.prompt_id == p.id]
                    print(f"Loaded attempts: Prompt {p.id} has {len(p[attempts])} attempts")
        if os.path.exists(folder+'/comparisons.jsonl'):
            with open(folder+'/comparisons.json', 'r') as f:
                initial_comparisons = [json.loads(line) for line in f.readlines()]
                for p in initial_population:
                    p.comparisons = [c for c in initial_comparisons if p.id in [c['prompt_a'], c['prompt_b']]]
                    print(f"Loaded comparisons: Prompt {p.id} has {len(p.comparisons)} comparisons")
    else:
        os.mkdir(folder)
    return initial_population, initial_attempts, initial_comparisons

def sep_norm_sort(string, sep1=',', sep2=';'):
    if hasattr(string, "split"):
        return [sorted([ss.strip().lower() for ss in s.split(sep1)]) for s in string.split(sep2)]
    return None            

def set_limits():
    #soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    #new_limit = min(hard, 512 * 1024 * 1024)  # can't go above current hard limit
    #resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))  # 5-second CPU limit

def exec_helper(code, queue, input_iterator):
    set_limits()

    f = io.StringIO()
    builtins = SAFE_BUILTINS.copy()
    builtins.update({
        "input": lambda: next(input_iterator)  # Use the next value from the inputs iterator
    })
    try:
        with redirect_stdout(f):
            exec(compile(code, "<string>", "exec"), {"__builtins__": builtins}, {})
        queue.put(f.getvalue().strip())
    except Exception as e:
        queue.put(f"Exception: {e}\n{traceback.format_exc()}")


def execute_code(raw_code: str, inputs=[]) -> str:
    if raw_code is None:
        return "Exception: No code provided."
    parts = raw_code.split('```')
    if len(parts) == 3:
        sanitized_code = '\n'.join(parts[1].split('\n')[1:])
    else:
        sanitized_code = raw_code
    sanitized_code = sanitized_code.encode().decode('unicode_escape')

    input_iterator = iter(inputs)  # Create an iterator from the inputs list

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=exec_helper, args=(sanitized_code, queue, input_iterator))
    p.start()
    p.join(6)  # 10 second timeout

    if p.is_alive():
        p.terminate()
        return "Exception: Process exceeded time limit."

    return queue.get() if not queue.empty() else "Exception: No output detected."

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
    

def str_to_type(type_str: str):
    return getattr(builtins, type_str, None)  


T = TypeVar("T")

def try_parse(val: Any, typ: Type[T]) -> T | None:
    if isinstance(val, typ if not get_origin(typ) else get_origin(typ)):
        return val
    try:
        parsed = ast.literal_eval(val) if isinstance(val, str) else val
        if isinstance(parsed, typ if not get_origin(typ) else get_origin(typ)):
            if get_origin(typ) and all(isinstance(item, get_args(typ)[0]) for item in parsed):
                return parsed
    except (ValueError, SyntaxError):
        pass
    return None
    
class imageurl(str): 
    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __repr__(self):
        return f"imageurl({self.value})"
    
if __name__ == "__main__":
    generic = list[int]
    val = "[9, 2]"
    assert try_parse(val, generic) == [9, 2]

    val_str = "['a', 'b']"
    generic_str = list[str]
    assert try_parse(val_str, generic_str) == ['a', 'b']

    val_list = ['a', 'b']
    assert try_parse(val_list, generic_str) == ['a', 'b']

if __name__ == "__main__":
    download_codecontests()