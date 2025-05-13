import builtins
import json
import os
import io
from contextlib import redirect_stdout
import traceback
import multiprocessing
import resource

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

def check_and_load_population(folder: str) -> list:
    """
    
    """
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
    """
    Normalization for connections outputs.
    """
    if hasattr(string, "split"):
        return [sorted([ss.strip().lower() for ss in s.split(sep1)]) for s in string.split(sep2)]
    return None            

def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))  # 5-second CPU limit

def exec_helper(code, queue, input_iterator):
    """
    Target function for code execution subprocess.
    """
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
    """
    Helper function that executes python code in a semi-safe environment (with safe built ins).
    """
    if raw_code is None:
        return "Exception: No code provided."
    
    # simple parse from markdown block
    parts = raw_code.split('```')
    if len(parts) == 3:
        sanitized_code = '\n'.join(parts[1].split('\n')[1:])
    else:
        sanitized_code = raw_code
    sanitized_code = sanitized_code.encode().decode('unicode_escape')


    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=exec_helper, args=(sanitized_code, queue, iter(inputs)))
    p.start()
    p.join(6)  # timeout

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

    
class imageurl(str): 
    def __init__(self, value):
        """
        Helper class to specify image generation tasks.
        """
        
        super().__init__(value)
        self.value = value

    def __repr__(self):
        return f"imageurl({self.value})"