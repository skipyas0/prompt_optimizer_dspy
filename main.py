import json

if __name__ == "__main__":
    # env setup
    import os
    import time
    stamp = 3953075#round(time.time() % 31536000)
    folder = f"runs/{stamp}"
    os.environ['RUN_FOLDER'] = folder
    
    from prompt import Prompt   
    if os.path.exists(folder):
        with open(folder+'/prompts.jsonl', 'r') as f:
            prompts = [json.loads(l) for l in f.readlines()]
            initial_population = [Prompt.from_json(p) for p in prompts]
    else:
        initial_population = []
        os.mkdir(folder)

    import dspy
    from dotenv import load_dotenv
    import utils
    from optimizer import Optimizer

    load_dotenv()
    api_key = os.getenv("API_KEY")
    
    vllm_port = os.getenv("VLLM_MY_PORT")
    if vllm_port:
        # self hosted model on cluster
        solve_lm = dspy.LM("hosted_vllm/ibnzterrell/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4", api_base=f"http://localhost:{vllm_port}/v1", api_key="EMPTY", cache=False)
        optim_lm = dspy.LM("hosted_vllm/ibnzterrell/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4", api_base=f"http://localhost:{vllm_port}/v1", api_key="EMPTY", cache=False)
    else:
        solve_lm = dspy.LM("gpt-4o-mini", api_key=api_key, cache=False)
        # use same as optim
        optim_lm = solve_lm
        # or define new one
        #solve_lm = dspy.LM("gpt-4o-mini", api_key=api_key, cache=False)

    ds = utils.load_data("seq.json")
    optim = Optimizer(ds, optim_lm, solve_lm)
    optim.begin(initial_population)
    optim.eval()
    
