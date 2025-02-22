import signatures
import dspy
import logging
import os
from prompt import Prompt
import utils
from population import Population
import matplotlib.pyplot as plt
import random
import Levenshtein
from data import Data

CREATIVE_TEMP = 0.75

N_SOLUTIONS = 1

POP_SIZE = 20
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
    def __init__(self, data: Data):
        logger.info(f"Settings: {POP_SIZE=}, {TOP_N=}, {ITER=}, {N_SOLUTIONS=}")
        self.data = data
        logger.info(f"Split lengths: train {len(self.data.train)}, dev {len(self.data.dev)}, test {len(self.data.test)}")
        self.start_gen = 1
        self.population = Population()
        self.all_prompts = Population()

    ### TOOLS FOR PROMPT GENERATION
    def lamarckian(self, hint: str = "", gen: int = 0, n: int = 1) -> None:
        """
        Task another component with access to the data to generate a new prompt.
        You may provide a short hint.
        This tool returns the resulting prompt and its score.

        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """
        module = dspy.ChainOfThought(signature=signatures.Lamarckian, temperature=CREATIVE_TEMP)
        for _ in range(n):
            prompt = module(examples=str(self.data.train), supervisor_hint=hint).prompt_proposal
            prompt_obj = Prompt(prompt, origin="lamarckian", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            self.population.update_tool_effectivity("lamarckian", score)
            logger.info(f"LAMARCKIAN generated prompt:\n {str(prompt_obj)}\nSCORE: {score}\nTools:{self.population.tool_effectivity}")

    def reflective(self, hint: str = "", gen: int = 0, n: int = 1) -> None:
        """
        Task another component to improve an underperfoming prompt.
        The prompt is automatically chosen from the worst quartile.
        Your subordinate will write a critique of the prompt and try to write a better one.
        You may provide a short hint.
        This tool returns the resulting prompt and its score.

        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """

        module = dspy.ChainOfThought(signature=signatures.Reflective, temperature=CREATIVE_TEMP)
        for _ in range(n):
            worst_quartile = self.population.quartile(4)
            original = random.choice(worst_quartile) if len(worst_quartile) > 1 else self.population.prompts[0]
            completion = original.get_completion(0)
            task = completion[0]
            reasoning = completion[1]
            completion = module(prompt=original.text, task_question=task.question, task_gold_answer=task.answer, reasoning=reasoning, supervisor_hint=hint)
            prompt = completion.prompt_proposal
            critique = completion.prompt_critique
            prompt_obj = Prompt(prompt, origin="reflective", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            logger.info(f"REFLECTIVE generated prompt:\n {str(prompt_obj)}\nSCORE: {score}.\nOriginal prompt:\n{original.text}\nCritique:\n{critique}\nTools:{self.population.tool_effectivity}")
            self.population.update_tool_effectivity("reflective", score)
    
    def iterative(self, hint: str = "", gen: int = 0, n: int = 1) -> None:
        """
        Task another component to create a new prompt based on several prompt+score examples.
        The examples are chosen automatically using roulette selection with the best prompts having the biggest chance of being featured.
        You may provide a short hint.
        This tool returns the resulting prompt and its score.

        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """

        module = dspy.ChainOfThought(signature=signatures.Iterative, temperature=CREATIVE_TEMP)
        for _ in range(n):
            examples = [p.prompt_and_perf() for p in self.population.select(5)]
            prompt = module(old_prompts=examples, supervisor_hint=hint).prompt_proposal
            prompt_obj = Prompt(prompt, origin="iterative", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            logger.info(f"ITERATIVE generated prompt:\n {str(prompt_obj)}\nSCORE: {score}.\nExamples:\n{examples}\nTools:{self.population.tool_effectivity}")
            self.population.update_tool_effectivity("iterative", score)

    def crossover(self, hint: str = "", gen: int = 0, n: int = 1) -> Prompt:
        """
        Task another component to create a new prompt using two diverse parent prompts.
        The prompts are chosen automatically.
        You may provide a short hint.
        This tool returns the resulting prompt and its score.

        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """
        
        module = dspy.ChainOfThought(signature=signatures.Crossover, temperature=CREATIVE_TEMP)
        for _ in range(n):
            best_quartile = self.population.quartile(1)
            prompt1 = random.choice(best_quartile) if len(best_quartile) > 1 else self.population.prompts[0]
            # prompt2 most distinct to prompt1
            prompt2 = sorted(self.population, key= lambda p: Levenshtein.distance(prompt1.text, p.text))[-1]
            prompts = [prompt1, prompt2]
            random.shuffle(prompts)
            prompt = module(prompts=tuple(prompts), supervisor_hint=hint).prompt_proposal
            prompt_obj = Prompt(prompt, origin="crossover", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            logger.info(f"CROSSOVER generated prompt:\n {str(prompt_obj)}\n SCORE: {score}\nfrom prompt1:\n{prompts[0]}\n and from prompt2:\n{prompts[1]}\nTools:{self.population.tool_effectivity}.")
            self.population.update_tool_effectivity("crossover", score)

    def __run(self):
        for step in range(self.start_gen+1,self.start_gen+ITER+1):
            self.all_prompts.set_update(self.population.prompts)
            n = self.population.purge_duplicates()
            self.reflective(gen=step, n = n)
            self.population.dump()

    def begin(self, initial_population: list[Prompt]=[]):
        if len(initial_population) < 1:
            self.lamarckian(n = POP_SIZE)
        else:
            self.population.add(initial_population)
            self.start_gen = max([p.gen for p in self.pop])
        self.pop.dump()

        logger.info("Starting optimization")
        self.__run()
        logger.info("Optimization done")
        all_prompts = Population(list(self.all_prompts))
        all_prompts.dump()

    def eval(self):
        logger.info("Starting final eval")
        by_gen = self.pop.evaluate_iterations(self.test)
        logger.info("Final eval done")

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



