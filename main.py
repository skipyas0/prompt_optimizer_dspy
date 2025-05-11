import os
import time
import json
if __name__ == "__main__":
    settings = {
        "batch_size": 3,
        "max_iters": 10,
        "lamarck_batch": 3, # 3 for code, 10 for seq and conn
        "pop_size": 10,
        "operator": "REFLECTIVE",
        "seeding_source": "PERSONAS",
        "dataset": "connections",
        "grading": "match_lists",
        "eval": "match_lists",
        "answer_type": "str",
        "do_optim": True,
        "do_eval": True,
        "run": "connections1_REFLECTIVE",
        "debug": False,
        "purge": "duplicates"
    }


    if settings["run"] is None:
        settings["run"] = f"{round(time.time() % 31536000)}"
    print("Settings:")
    print(settings)
    folder = f"runs/{settings['run']}"
    print(f"Running in {folder}")
    os.environ['RUN_FOLDER'] = folder
    if settings["debug"]:
        os.environ["DEBUG"] = ""

    import utils
    initial_population, initial_attempts, initial_comparisons = utils.check_and_load_population(folder)
    with open(f"{folder}/settings.json", "w+") as f:
        json.dump(settings, f)

    from optimizer import Optimizer
    from data import Data
    import grading

    grading_function = getattr(grading, settings["grading"]) if settings["grading"] else None
    eval_function = getattr(grading, settings["eval"]) if settings["eval"] else None
    data = Data.from_json(f"datasets/{settings['dataset']}.json", grading_function, eval_function, settings["answer_type"])
    data.update_attempts(initial_attempts)

    optim = Optimizer(data, settings)
    optim.begin(initial_population)

    
    print("Finished run with settings:")
    print(settings)
