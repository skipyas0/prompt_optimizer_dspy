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
    lm = dspy.LM("gpt-4o-mini", api_key=api_key, cache = False)
    dspy.configure(lm=lm)
    
    ds = utils.load_data("seq.json")
    optim = Optimizer(ds)
    optim.begin()
    optim.eval()
    
