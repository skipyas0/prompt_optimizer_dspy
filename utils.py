import builtins
from datasets import load_dataset
import json
import dspy
import os
from typing import Literal
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

def load_data(path):
    with open(f"{path}", "r") as f:
        data = json.load(f)
    def examplify(s):
        ret = dspy.Example(question=s["question"], answer=s["answer"])
        ret.with_inputs("question")
        return ret
    
    data = map(examplify, data)
    return list(data)


def load_gsm8k_server():
    ds = load_dataset('openai/gsm8k', 'main', split='train').select(range(15))

    def map_gsm8k(example):
        example['question'] = example['question']
        example['answer'] = example['answer'].split('####')[1].replace('\xa0', '').strip()
        return {'question': example['question'], 'answer': example['answer']}
    
    ds = ds.map(map_gsm8k, remove_columns=ds.column_names, load_from_cache_file=False).to_list()
    def examplify(s):
        ret = dspy.Example(question=s["question"], answer=s["answer"])
        ret.with_inputs("question")
        return ret
    
    ds = map(examplify, ds)
    return list(ds)


def format_examples(examples: list[dspy.Example]) -> str:
    TEMPLATE = "Question: {q}\nAnswer: {a}"
    example_strings = [TEMPLATE.format(q=e.question, a=e.answer) for e in examples]
    return "\n".join(example_strings)

def get_lm(lm_use: Literal["OPTIM", "SOLVE"], uni: bool = True) -> dspy.LM:
    vllm_port = os.getenv("VLLM_MY_PORT")
    
    if lm_use == "OPTIM":
        lm = os.environ["OPTIM_LM"]
        op_type = os.environ["OPTIM_OP"]
    else:
        lm = os.environ["SOLVE_LM"]
        op_type = "SOLVE"

    if uni:
        op_type = "UNIVERSAL"

    if "gpt" in lm:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ[f"API_KEY_{op_type}"]
        lm = dspy.LM(lm, api_key=api_key, cache=False)
    elif vllm_port:
        lm = dspy.LM(lm, api_base=f"http://localhost:{vllm_port}/v1", api_key="EMPTY", cache=False)
    else:
        raise ValueError("No valid model config for lm {lm} and no vllm port")
    return lm

def check_and_load_population(folder: str) -> list:
    initial_population = []
    if os.path.exists(folder):
        if os.path.exists(folder+'/prompts.jsonl'):
            from prompt import Prompt   
            with open(folder+'/prompts.jsonl', 'r') as f:
                prompts = [json.loads(l) for l in f.readlines()]
                initial_population = [Prompt.from_json(p) for p in prompts]
    else:
        os.mkdir(folder)
    return initial_population

def deseparate_into_lists(string, sep1=',', sep2=';'):
    if hasattr(string, "split"):
        return [s.split(sep1) for s in string.split(sep2)]
    return None            
    

def recursive_string_normalize(inp: str | list):
    if type(inp) == list:
        return [recursive_string_normalize(subinp) for subinp in inp]
    if type(inp) == str:
        return inp.strip().lower()
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

antonymum = dspy.Predict("phrase: str, phrase_context: str -> opposite_meaning_phrase_in_context: str")
twist = dspy.Predict("phrase: str, phrase_context: str -> phrase_with_unexpected_twist: str")

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
    

def str_to_type(type_str: str):
    return getattr(builtins, type_str, None)  



T = TypeVar("T")



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
