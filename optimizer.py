import logging
import os
from prompt import Prompt
from population import Population
import matplotlib.pyplot as plt
import random
import Levenshtein
from data import Data
import my_signatures as sig
from model_api import model


N_SOLUTIONS = 1

POP_SIZE = 4  # 20
ITER = 3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Optimizer:
    def __init__(self, data: Data):
        logger.info(f"Settings: {POP_SIZE=}, {ITER=}, {N_SOLUTIONS=}")
        self.data = data
        logger.info(
            f"Split lengths: train {len(self.data.train)}, dev {len(self.data.dev)}, test {len(self.data.test)}"
        )
        self.start_gen = 0
        self.population = Population([])
        self.all_prompts = Population([])

        self.operators = {
            "LAMARCKIAN": self.lamarckian,
            "REFLECTIVE": self.reflective,
            "ITERATIVE": self.iterative,
            "CROSSOVER": self.crossover,
            "MUTATION": self.mutation,
        }
        op_type = os.environ["OPTIM_OP"]
        self.op = self.operators[op_type]
        with open("datasets/values.csv", "r") as f:
            user_values = [line.split(',')[0] for line in f.read().split('\n')]

        self.random_value_focus = lambda: random.sample(user_values,3)

    ### TOOLS FOR PROMPT GENERATION
    def lamarckian(self, gen: int = 0, n: int = 1) -> None:
        """
        Task another component with access to the data to generate a new prompt.
        You may provide a short hint.

        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """
        N_EXAMPLES = 5

        for _ in range(n):
            examples = self.data.select("train", N_EXAMPLES)
            _, completion = model.chain_of_thought(
                sig.lamarckian6, task_examples=examples, focus=self.random_value_focus()
            )
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="lamarckian", gen=gen
            )
            score = self.data.eval_on_split(prompt_obj)
            self.population.add(prompt_obj)
            self.population.update_tool_effectivity("lamarckian", score)
            logger.info(
                f"LAMARCKIAN generated prompt:\n {str(prompt_obj)}\nSCORE: {score}\nTools:{self.population.tool_effectivity}"
            )

    def reflective(self, gen: int = 0, n: int = 1) -> None:
        """ """

        for _ in range(n):
            solution = None
            while not solution:
                original = random.choice(self.population.prompts)
                solution = original.get_completion(0)
                logger.warning(
                    f"Got past solution {solution} in reflective, prompt {str(original)}"
                )
            task = solution[0]
            reasoning = solution[1]
            _, completion = model.chain_of_thought(
                signature=sig.reflective2,
                original_prompt=original.text,
                task_question=task,
                solution=reasoning,
            )
            prompt_obj = Prompt(completion['prompt_proposal'], origin="reflective", gen=gen)
            score = self.data.eval_on_split(prompt_obj)
            self.population.add(prompt_obj)
            logger.info(
                f"REFLECTIVE generated prompt:\n {str(prompt_obj)}\nSCORE: {score}.\nOriginal prompt:\n{original.text}\nTools:{self.population.tool_effectivity}"
            )
            self.population.update_tool_effectivity("reflective", score)

    def iterative(self, gen: int = 0, n: int = 1) -> None:
        """ """
        for _ in range(n):
            examples = [p.prompt_and_perf() for p in self.population.select(5)]
            examples = sorted(examples, key=lambda t: t[1])
            _, completion = model.chain_of_thought(sig.iterative, old_prompts=examples)
            prompt_obj = Prompt(completion["prompt_proposal"], origin="iterative", gen=gen)
            score = self.data.eval_on_split(prompt_obj)
            self.population.add(prompt_obj)
            logger.info(
                f"ITERATIVE generated prompt:\n {str(prompt_obj)}\nSCORE: {score}.\nExamples:\n{examples}\nTools:{self.population.tool_effectivity}"
            )
            self.population.update_tool_effectivity("iterative", score)

    def crossover(self, gen: int = 0, n: int = 1) -> Prompt:
        """ """
        for _ in range(n):
            best_quartile = self.population.quartile(1)
            prompt1 = (
                random.choice(best_quartile)
                if len(best_quartile) > 1
                else self.population.prompts[0]
            )
            # prompt2 most distinct to prompt1
            prompt2 = sorted(
                self.population,
                key=lambda p: Levenshtein.distance(prompt1.text, p.text),
            )[-1]
            prompts = [prompt1, prompt2]
            random.shuffle(prompts)
            completion = model.chain_of_thought(
                signature=sig.crossover,
                prompt_a=prompts[0].prompt_and_perf(),
                prompt_b=prompts[1].prompt_and_perf(),
            )
            prompt_obj = Prompt(completion["prompt_proposal"], origin="crossover", gen=gen)
            score = self.data.eval_on_split(prompt_obj)
            self.population.add(prompt_obj)
            logger.info(
                f"CROSSOVER generated prompt:\n {str(prompt_obj)}\n SCORE: {score}\nfrom prompt1:\n{prompts[0]}\n and from prompt2:\n{prompts[1]}\nTools:{self.population.tool_effectivity}."
            )
            self.population.update_tool_effectivity("crossover", score)

    def mutation(self, gen: int = 0, n: int = 1):
        """ """
        for _ in range(n):
            input_prompt = random.choice(self.population.prompts)
            completion = model.chain_of_thought(sig.mutation, input_prompt=input_prompt)
            prompt_obj = Prompt(completion['prompt_proposal'], origin="mutation", gen=gen)
            score = self.data.eval_on_split(prompt_obj)
            self.population.add(prompt_obj)
            logger.info(
                f"MUTATION generated prompt:\n {str(prompt_obj)}\n SCORE: {score}\nfrom prompt:{input_prompt}\nTools:{self.population.tool_effectivity}."
            )
            self.population.update_tool_effectivity("mutation", score)

    def __run(self):
        for step in range(self.start_gen + 1, self.start_gen + ITER + 1):
            print("step", step)
            self.all_prompts.set_update(self.population.prompts)
            self.all_prompts.dump()
            #n = self.population.purge_duplicates()
            self.op(gen=step, n=1)
            self.population.dump(gen=step)
        # add last generation
        self.all_prompts.set_update(self.population.prompts)

    def begin(self, initial_population: list[Prompt] = []):
        if len(initial_population) > 0:
            self.all_prompts.set_update(initial_population)
            active = [p for p in self.all_prompts if p.active]
            # score prompts with uninitialized dev scores
            _ = [
                self.data.eval_on_split(p) for p in active if p.get_score("dev") == -1.0
            ]
            self.population.add(active)
            self.start_gen = max([p.gen for p in self.population])
        if len(initial_population) < POP_SIZE:
            remaining = POP_SIZE - len(initial_population)
            self.lamarckian(n=remaining)
            self.all_prompts.set_update(self.population.prompts)
        self.population.dump(gen=0)
        self.all_prompts.dump()

        logger.info("Starting optimization")
        self.__run()
        logger.info("Optimization done")
        self.all_prompts.dump()

    def eval(self):
        logger.info("Starting final eval")
        by_gen = self.all_prompts.evaluate_iterations(self.data)
        logger.info("Final eval done")
        self.all_prompts.dump()

        x = list(range(len(by_gen)))
        y_avg = [sum(g) / len(g) for g in by_gen]
        y_max = [max(g) for g in by_gen]
        logger.info(
            f"Evaluation stats:\nAvg: {' '.join(map(str, y_avg))},\nMax: {' '.join(map(str, y_max))}"
        )
        plt.figure()
        plt.title("OPRO-like Hill-Climber")
        plt.plot(x, y_avg, color="blue", label="Average")
        plt.plot(x, y_max, color="red", label="Max")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Average score")
        plt.savefig(f"{os.getenv('RUN_FOLDER')}/plt.svg")
