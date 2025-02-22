import json
import os
import time
if __name__ == "__main__":
    # env setup
    os.environ["OPTIM_LM"] = "gpt-4o"
    os.environ["SOLVE_LM"] = "gpt-4o-mini"
    stamp = round(time.time() % 31536000)
    folder = f"runs/{stamp}"
    os.environ['RUN_FOLDER'] = folder
    
    from prompt import Prompt   
    if os.path.exists(folder):
        with open(folder+'/all_prompts.jsonl', 'r') as f:
            prompts = [json.loads(l) for l in f.readlines()]
            initial_population = [Prompt.from_json(p) for p in prompts]
    else:
        initial_population = []
        os.mkdir(folder)

    import utils
    from optimizer import Optimizer
    from data import Data
    
    optim_lm = utils.get_lm(os.environ["OPTIM_LM"])
    data = Data.from_json("seq.json")
    optim = Optimizer(data)
    optim.begin(initial_population)
    optim.eval()
    
