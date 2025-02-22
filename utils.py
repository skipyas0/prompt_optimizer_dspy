from datasets import load_dataset
import json
import dspy
import os

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

def get_lm(lm: str) -> dspy.LM:
    vllm_port = os.getenv("VLLM_MY_PORT")
    if "gpt" in lm:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("API_KEY")
        lm = dspy.LM(lm, api_key=api_key, cache=False)
    elif vllm_port:
        lm = dspy.LM(lm, api_base=f"http://localhost:{vllm_port}/v1", api_key="EMPTY", cache=False)
    else:
        raise ValueError("No valid model config for lm {lm} and no vllm port")
    return lm

if __name__ == "__main__":
    download_gsm8k()