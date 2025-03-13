import os
import dspy

if __name__ == "__main__":
    # env setup
    stamp = "gen10s"
    folder = f"runs/{stamp}"

    os.environ["SOLVE_LM"] = "gpt-4o-mini"
    os.environ['RUN_FOLDER'] = folder
    
    import utils
    initial_population = utils.check_and_load_population(folder)
    from population import Population
    from data import Data
    pop = Population(initial_population)
    data = Data.from_json("datasets/sequence_bench_quad_alt_rec_mod.json")
    
    pop.evaluate_iterations(data)
    pop.dump()

