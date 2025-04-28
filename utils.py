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


def check_and_load_population(folder: str) -> list:
    initial_population = []
    if os.path.exists(folder):
        if os.path.exists(folder+'/prompts.jsonl'):
            from prompt import Prompt   
            with open(folder+'/prompts.jsonl', 'r') as f:
                prompts = [json.loads(line) for line in f.readlines()]
                initial_population = [Prompt.from_json(p) for p in prompts]
    else:
        os.mkdir(folder)
    return initial_population

def sep_norm_sort(string, sep1=',', sep2=';'):
    if hasattr(string, "split"):
        return [sorted([ss.strip().lower() for ss in s.split(sep1)]) for s in string.split(sep2)]
    return None            
    


def execute_code(raw_code: str) -> str:
    parts = raw_code.split('```')
    if len(parts) == 3:
        sanitized_code = '\n'.join(parts[1].split('\n')[1:])
    else:
        sanitized_code = raw_code
    sanitized_code = sanitized_code.encode().decode('unicode_escape')

    
    def set_limits():
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024))  # 3GB memory limit
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))  # 5-second CPU limit

    def exec_helper(code, queue):
        set_limits() 

        f = io.StringIO()
        local_vars = {"input": lambda: None}  

        try:
            with redirect_stdout(f):  
                exec(compile(code, "<string>", "exec"), {"__builtins__": SAFE_BUILTINS}, local_vars)
            queue.put(f.getvalue().strip())  
        except Exception as e:
            queue.put(f"Exception: {e}\n{traceback.format_exc()}")

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=exec_helper, args=(sanitized_code, queue))
    p.start()
    p.join(6)  # 10 second timeout

    if p.is_alive():
        p.terminate()
        return "Process exceeded time limit."

    return queue.get() if not queue.empty() else "No output detected."

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
    
if __name__ == "__main__":
    generic = list[int]
    val = "[9, 2]"
    assert try_parse(val, generic) == [9, 2]

    val_str = "['a', 'b']"
    generic_str = list[str]
    assert try_parse(val_str, generic_str) == ['a', 'b']

    val_list = ['a', 'b']
    assert try_parse(val_list, generic_str) == ['a', 'b']
