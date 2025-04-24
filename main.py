import os
import time

if __name__ == "__main__":
    # env setup
    stamp = round(time.time() % 31536000)
    folder = f"runs/{stamp}"


    os.environ["OPTIM_OP"] = "REFLECTIVE"
    os.environ['RUN_FOLDER'] = folder
    
    import utils
    initial_population = utils.check_and_load_population(folder)

    from optimizer import Optimizer
    from data import Data

    data = Data.from_json("archive/seq.json", "int")
    optim = Optimizer(data)
    optim.begin(initial_population)
    optim.eval()
    
