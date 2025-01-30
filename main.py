import dspy
from dotenv import load_dotenv
import os
import utils
import signatures
import time
import matplotlib.pyplot as plt
import logging
import re
CREATIVE_TEMP = 0.9
SOLVE_TEMP = 0.0
N_SOLUTIONS = 3

POP_SIZE = 10
TOP_N = 5
ITER = 10

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




def score_prompt(prompt: str, batch: list[dspy.Example], solve: dspy.Module) -> list[float]:
    batch_score = 0.0
    for i, example in enumerate(batch):
        question = prompt.format(example.question)
        solutions = solve(question=question).completions
        example = float(example.answer)
        avg_score_on_sample = 0.0
        for comp_idx in range(N_SOLUTIONS):
            reasoning = solutions.reasoning[comp_idx]
            solution = solutions.answer[comp_idx]
            try:
                solution = float(solution)
                grade = 1.0 if solution==example else 0.0
            except ValueError:
                logger.warning(f"Couldn't convert '{solution}' to float")
                grade = 0.0
            avg_score_on_sample += grade
            logger.debug(f"Grading problem {i+1}\nReasoning:{reasoning}\nSolution:{solution}\t|\tGold:{example}\nPass:{grade}\n")
        batch_score += avg_score_on_sample / N_SOLUTIONS
    return batch_score / len(batch)

def rank(prompts: list[str], batch: list[dspy.Example], solve: dspy.Module) -> list[tuple[str, float]]:
    prompts_and_scores = []
    for prompt in prompts:
        batch_score = 0.0 if len(re.findall("{.*?}", prompt)) != 1 else score_prompt(prompt, batch, solve)
        prompts_and_scores.append((prompt, batch_score))
        logger.info(f"Registered score {batch_score} for prompt\n{prompt}")
    prompts_and_scores.sort(key=lambda tup: tup[1], reverse=True)
    return prompts_and_scores

if __name__ == "__main__":
    # env setup
    stamp = round(time.time() % 31536000)
    folder = f"run{stamp}"
    os.mkdir(folder)
    load_dotenv()
    api_key = os.getenv("API_KEY")
    lm = dspy.LM("gpt-4o-mini", api_key=api_key, cache = False)
    dspy.configure(lm=lm)
    top_from_each_gen = []

    # logging setup
    debug_handler = logging.FileHandler(f"{folder}/debug.log")
    debug_handler.setLevel(logging.DEBUG)
    info_handler = logging.FileHandler(f"{folder}/info.log")
    info_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    debug_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)


    
    # load splits
    ds = utils.load_data("seq.json")
    train, dev, test = ds[:5], ds[5:10], ds[10:]

    # setup llm callables
    induce = dspy.ChainOfThought(signature=signatures.InstructionInductor, temperature=CREATIVE_TEMP, n=POP_SIZE)
    iterate = dspy.ChainOfThought(signature=signatures.OptimizerIterator, temperature=CREATIVE_TEMP, n=POP_SIZE-TOP_N)
    solve = dspy.ChainOfThought("question: str -> answer: float", temperature=SOLVE_TEMP, n=N_SOLUTIONS)
    
    
    # first induce initial population
    examples = utils.format_examples(train)
    completions = induce(examples=examples).completions
    pop = list(map(lambda s: s[0]+"{}"+s[1], zip(completions.prefix_prompt, completions.suffix_prompt)))
    logger.info("Instruction Induction complete")

    # optimization process
    for i in range(ITER):
        logger.info(f"Starting iteration {i+1}")

        # get best performing prompts
        prompts_and_scores = rank(pop, dev, solve)
        scores = [t[1] for t in prompts_and_scores]
        best_with_scores = prompts_and_scores[:TOP_N]

        # save stats and prompts
        avg_score, max_score = sum(scores) / len(scores), max(scores)
        logger.info(f"Iteration {i+1} average score: {avg_score}, max score: {max_score}\n")
        utils.dump_prompts(prompts_and_scores, f"{folder}/prompts_{i+1}.jsonl")
        top_from_each_gen.append(best_with_scores)

        # generate new prompts based on the best ones
        best = [t[0] for t in best_with_scores]
        new = iterate(old_prompts=best).completions.prompt_proposal

        #integrate them to population
        pop = best + new

    scores = []
    logger.info("Optimization steps done")
    for i, g in enumerate(top_from_each_gen):
        prompts_and_scores = rank([t[0] for t in g], test, solve)
        scores.append(sum([t[1] for t in prompts_and_scores])/len(g))


    plt.figure()
    plt.title("OPRO-like Hill-Climber")
    x = list(range(1,len(scores)+1))
    plt.plot(x, scores)
    plt.xlabel("Iteration")
    plt.ylabel("Average score")
    plt.savefig(f"{folder}/plt.svg")
    plt.show()

