import os
import time
import json

# done: conn, seq
# refl, fb, iter, para
ds = "sequences"
op = "feedback"
ix = 2



answer_type = "int" if ds == "sequences" else "str"
eval_f = "match_lists" if ds == "connections" else "exact_match_float" if ds == "sequences" else "test_code"
grading = None if op == "feedback" else eval_f

if __name__ == "__main__":
    settings = {
        "batch_size": 3,
        "max_iters": 0,
        "lamarck_batch": 3, # 3 for code, 10 for seq and conn
        "pop_size": 10,
        "operator": op.upper(),
        "seeding_source": "PERSONAS",
        "dataset": ds,
        "grading": grading,
        "eval": eval_f,
        "answer_type": answer_type,
        "do_optim": False,
        "do_eval": True,
        "run": f"{ds}{ix}_{op.upper()}",
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
