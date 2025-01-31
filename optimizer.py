import signatures
import dspy
import logging
import os
from prompt import Prompt
import utils
from population import Population
import matplotlib.pyplot as plt

CREATIVE_TEMP = 0.9
SOLVE_TEMP = 0.0
N_SOLUTIONS = 1

POP_SIZE = 5
TOP_N = 3
ITER = 100

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class Optimizer:
    def __init__(self, data: list[dspy.Example], optim_lm: dspy.LM, solve_lm: dspy.LM):
        ss = len(data) // 3
        self.train, self.dev, self.test = data[:ss], data[ss:2*ss], data[2*ss:]
        logger.info(f"Split lengths: train {len(self.train)}, dev {len(self.dev)}, test {len(self.test)}")
        logger.info("Optimizer model" + optim_lm.model)
        logger.info("Solver model" + solve_lm.model)
        induce_module = dspy.ChainOfThought(signature=signatures.InstructionInductor, temperature=CREATIVE_TEMP, n=POP_SIZE)
        iterate_module = dspy.ChainOfThought(signature=signatures.OptimizerIterator, temperature=CREATIVE_TEMP, n=POP_SIZE)
        solve_module = dspy.ChainOfThought("question: str -> answer: float", temperature=SOLVE_TEMP, n=N_SOLUTIONS)

        def induce(examples):
            with dspy.context(lm=optim_lm):
                ret = induce_module(examples=examples)
            return ret
        self.induce = induce
        
        def iterate(old_prompts):
            with dspy.context(lm=optim_lm):
                ret = iterate_module(old_prompts=old_prompts)
            return ret
        self.iterate = iterate

        def solve(question):
            with dspy.context(lm=solve_lm):
                ret = solve_module(question=question)
            return ret
        self.solve = solve

        self.pop = Population([], self.solve)

    def __induce(self, examples: str) -> list[str]:
        logger.info("Starting instruction induction")
        completions = self.induce(examples=examples).completions
        pop = list(map(lambda s: Prompt(s[0]+'{}'+s[1],0), zip(completions.prefix_prompt, completions.suffix_prompt)))
        logger.info("Instruction Induction complete")
        return pop

    def __step(self, i: int):
        logger.info(f"Starting iteration {i}")
        best = self.pop.select(TOP_N, self.dev)

        # save stats and prompts
        avg_score, max_score = self.pop.stats
        logger.info(f"Iteration {i} average score: {avg_score}, max score: {max_score}")
        self.pop.dump()

        # generate new prompts based on the best ones
        best_str = [str(p) for p in best]
        new = [Prompt(text, i) for text in self.iterate(old_prompts=best_str).completions.prompt_proposal]
        self.pop.add(new)

    def __run(self):
        for i in range(1,ITER+1):
            self.__step(i)

    def begin(self, initial_population: list[Prompt]=[]):
        if len(initial_population) < 1:
            text_examples = utils.format_examples(self.train)
            self.pop.add(self.__induce(text_examples))
        else:
            self.pop.add(initial_population)

        logger.info("Starting optimization")
        self.__run()
        logger.info("Optimization done")

    def eval(self):
        logger.info("Starting final eval")
        by_gen = self.pop.evaluate_iterations(self.test)
        logger.info("Final eval done")
        self.pop.dump()

        x = list(range(len(by_gen)))
        y_avg = [sum(g)/len(g) for g in by_gen]
        y_max = [max(g) for g in by_gen]
        logger.info(f"Evaluation stats:\nAvg: {' '.join(map(str,y_avg))},\nMax: {' '.join(map(str,y_max))}")
        plt.figure()
        plt.title("OPRO-like Hill-Climber")
        plt.plot(x, y_avg, color="blue", label="Average")
        plt.plot(x, y_max, color="red", label="Max")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Average score")
        plt.savefig(f"{os.getenv('RUN_FOLDER')}/plt.svg")



