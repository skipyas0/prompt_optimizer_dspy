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

POP_SIZE = 15
TOP_N = 5
ITER = 0#10

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class Optimizer:
    def __init__(self, data: list[dspy.Example], optim_lm: dspy.LM, solve_lm: dspy.LM):
        logger.info(f"Settings: {POP_SIZE=}, {TOP_N=}, {ITER=}, {N_SOLUTIONS=},{CREATIVE_TEMP=}, {SOLVE_TEMP=}")
        ss = len(data) // 3
        self.train, self.dev, self.test = data[:ss], data[ss:2*ss], data[2*ss:]
        logger.info(f"Split lengths: train {len(self.train)}, dev {len(self.dev)}, test {len(self.test)}")
        logger.info("Optimizer model " + optim_lm.model)
        logger.info("Solver model " + solve_lm.model)
        self.start_gen = 1
        induce_module = dspy.ChainOfThought(signature=signatures.InstructionInductor, temperature=CREATIVE_TEMP, n=POP_SIZE)
        iterate_module = dspy.ChainOfThought(signature=signatures.OptimizerIterator, temperature=CREATIVE_TEMP, n=POP_SIZE)
        solve_module = dspy.ChainOfThought("question: str -> answer: float", temperature=SOLVE_TEMP, n=N_SOLUTIONS)

        def induce(examples):
            try:
                with dspy.context(lm=optim_lm):
                    ret = induce_module(examples=examples)
                return ret
            except ValueError as e:
                logger.error(f"Model/COT error '{str(e)}' on induce, repeating")
                return induce(examples)
        self.induce = induce
        
        def iterate(old_prompts):
            try:
                with dspy.context(lm=optim_lm):
                    ret = iterate_module(old_prompts=old_prompts)
                return ret
            except ValueError as e:
                logger.error(f"Model/COT error '{str(e)}' on iterate, repeating")
                return iterate(old_prompts)
            
        self.iterate = iterate

        def solve(question):
            try:
                with dspy.context(lm=solve_lm):
                    ret = solve_module(question=question)
                return ret
            except ValueError as e:
                logger.error(f"Model/COT error '{str(e)}' on solve, aborting solve")
                return None
            
        self.solve = solve

        self.pop = Population([], self.solve)

    def __induce(self, examples: str) -> list[str]:
        logger.info("Starting instruction induction")
        completions = self.induce(examples=examples).completions
        pop = list(map(lambda s: Prompt(s[0], s[1],0), zip(completions.prefix_prompt, completions.suffix_prompt)))
        logger.info("Instruction Induction complete")
        return pop

    def __step(self, i: int):
        logger.info(f"Starting iteration {i}")
        best = self.pop.select(TOP_N, self.dev)

        # save stats and prompts
        avg_score, max_score = self.pop.stats
        logger.info(f"Iteration {i} average score: {avg_score}, max score: {max_score}")

        # generate new prompts based on the best ones
        best_str = [p.prompt_and_perf() for p in best]
        logger.info("Generating new prompts from:\n"+'\n'.join([f"Prompt: {p[0]}, score: {p[1]}" for p in best_str]))
        new = [Prompt(text, "", i) for text in self.iterate(old_prompts=best_str).completions.prompt_proposal]
        self.pop.add(new)
        self.pop.dump()

    def __run(self):
        for i in range(self.start_gen,self.start_gen+ITER+1):
            self.__step(i)

    def begin(self, initial_population: list[Prompt]=[]):
        if len(initial_population) < 1:
            text_examples = utils.format_examples(self.train)
            self.pop.add(self.__induce(text_examples))
        else:
            self.pop.add(initial_population)
            self.start_gen = max([p.gen for p in self.pop])
        self.pop.dump()

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



