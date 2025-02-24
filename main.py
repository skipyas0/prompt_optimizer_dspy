import os
import time
import dspy

if __name__ == "__main__":
    # env setup
    stamp = "experiment-reflective"#round(time.time() % 31536000)
    folder = f"runs/{stamp}"

    os.environ["OPTIM_LM"] = "gpt-4o"
    os.environ["SOLVE_LM"] = "gpt-4o-mini"
    os.environ["OPTIM_OP"] = "REFLECTIVE"
    os.environ['RUN_FOLDER'] = folder
    
    import utils
    initial_population = utils.check_and_load_population(folder)

    from optimizer import Optimizer
    from data import Data

    optim_lm = utils.get_lm("OPTIM")
    dspy.configure(lm=optim_lm)

    data = Data.from_json("sequence_bench_quad_alt_rec_mod.json")
    optim = Optimizer(data)
    optim.begin(initial_population)
    optim.eval()
    
