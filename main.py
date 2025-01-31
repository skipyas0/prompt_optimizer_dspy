if __name__ == "__main__":
    # env setup
    import os
    import time
    stamp = round(time.time() % 31536000)
    folder = f"runs/{stamp}"
    os.environ["RUN_FOLDER"] = folder
    os.mkdir(folder)

    import dspy
    from dotenv import load_dotenv
    import utils
    from optimizer import Optimizer

    load_dotenv()
    api_key = os.getenv("API_KEY")
    optim_lm = dspy.LM("gpt-4o-mini", api_key=api_key, cache=False)

    vllm_port = os.getenv("VLLM_MY_PORT")
    if vllm_port:
        # self hosted model on cluster
        solve_lm = dspy.LM("hosted_vllm/CohereForAI/aya-expanse-8b", api_base=f"http://localhost:{port}/v1", api_key="EMPTY", cache=False)
    else:
        # use same as optim
        solve_lm = optim_lm
        # or define new one
        #solve_lm = dspy.LM("gpt-4o-mini", api_key=api_key, cache=False)

    ds = utils.load_data("seq.json")
    optim = Optimizer(ds, optim_lm, solve_lm)
    optim.begin()
    optim.eval()
    
