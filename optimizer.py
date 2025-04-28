import logging
import os
from prompt import Prompt
from population import Population
import matplotlib.pyplot as plt
import random
import Levenshtein
from data import Data, Example
import my_signatures as sig
from model_api import model


N_SOLUTIONS = 1

POP_SIZE = 3  # 20
ITER = 3
BATCH_SIZE = 1
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
            "MUTATION": self.paraphrase,
            "FEEDBACK": self.feedback,
        }
        op_type = os.environ["OPTIM_OP"]
        self.op = self.operators[op_type]
        with open("datasets/values.csv", "r") as f:
            user_values = [line.split(",")[0] for line in f.read().split("\n")]

        self.random_value_focus = lambda: random.sample(user_values, 3)

        self.start_batch = self.data.get_batch("dev", n=BATCH_SIZE)

    ### TOOLS FOR PROMPT GENERATION
    def lamarckian(self, gen: int = 0, n: int = 1) -> list[Prompt]:
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
        prompts = []
        for _ in range(n):
            examples = [e.qa_dict() for e in self.data.select("train", N_EXAMPLES)]
            _, completion = model.chain_of_thought(
                sig.lamarckian8, task_examples=examples, focus=self.random_value_focus()
            )
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="lamarckian", gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(f"LAMARCKIAN generated prompt:\n {str(prompt_obj)}\n")
            prompts.append(prompt_obj)
        return prompts

    def reflective(self, gen: int = 0, n: int = 1) -> list[Prompt]:
        """ """
        prompts = []
        for _ in range(n):
            original = random.choice(self.population.prompts)
            solutions = self.data.get_all_prompt_attemps(original.id, wanted_grade=0)
            if len(solutions) == 0:
                solutions = self.data.get_all_prompt_attemps(original.id)
            solution = random.choice(solutions) if len(solutions) > 0 else None
            if solution is None:
                raise ValueError(f"No past solution for prompt id {original.id}")

            task = solution[0]
            reasoning = solution[1]
            _, completion = model.chain_of_thought(
                signature=sig.reflective2,
                original_prompt=original.text,
                task_question=task,
                solution=reasoning,
            )
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="reflective", gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(
                f"REFLECTIVE generated prompt:\n {str(prompt_obj)}\nOriginal prompt:\n{original.text}\n"
            )
            prompts.append(prompt_obj)
        return prompts

    def feedback(self, gen: int = 0, n: int = 1) -> list[Prompt]:
        """ """
        prompts = []
        for _ in range(n):
            original = random.choice(self.population.prompts)
            comparisons = []
            for comp in original.comparisons:
                winner = (
                    comp["prompt_a"]
                    if comp["verdict"] == "prompt_a"
                    else comp["prompt_b"]
                )
                other_id = (
                    comp["prompt_a"]
                    if original.id != comp["prompt_a"]
                    else comp["prompt_b"]
                )
                other = self.all_prompts.get_by_id(other_id)
                comparisons.append(
                    {
                        "original_won": winner == original.id,
                        "other_prompt": other.text,
                        "comparison": comp["prompt_comp"],
                    }
                )
            _, completion = model.chain_of_thought(
                signature=sig.feedback,
                base_prompt=original.text,
                comparisons=comparisons,
            )
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="reflective", gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(
                f"REFLECTIVE generated prompt:\n {str(prompt_obj)}\nOriginal prompt:\n{original.text}\n"
            )
            prompts.append(prompt_obj)
        return prompts

    def iterative(self, gen: int = 0, n: int = 1) -> list[Prompt]:
        """ """
        prompts = []
        for _ in range(n):
            examples = [p.prompt_and_perf() for p in self.population.select(5)]
            examples = sorted(examples, key=lambda t: t[1])
            _, completion = model.chain_of_thought(sig.iterative, old_prompts=examples)
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="iterative", gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(
                f"ITERATIVE generated prompt:\n {str(prompt_obj)}\nExamples:\n{examples}\n"
            )
            prompts.append(prompt_obj)
        return prompts

    def crossover(self, gen: int = 0, n: int = 1) -> list[Prompt]:
        """ """
        prompts = []
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
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="crossover", gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(
                f"CROSSOVER generated prompt:\n {str(prompt_obj)}\n from prompt1:\n{prompts[0]}\n and from prompt2:\n{prompts[1]}\n."
            )
            prompts.append(prompt_obj)
        return prompts

    def paraphrase(self, gen: int = 0, n: int = 1):
        """ """
        prompts = []
        for _ in range(n):
            input_prompt = random.choice(self.population.prompts)
            completion = model.chain_of_thought(
                sig.paraphrase, input_prompt=input_prompt
            )
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="paraphrase", gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(
                f"MUTATION generated prompt:\n {str(prompt_obj)}\n from prompt:{input_prompt}\n."
            )
            prompts.append(prompt_obj)
        return prompts

    def eval_and_sort(
        self, prompts: list[Prompt], batch: list[Example], target_pop="active"
    ):
        split = batch[0].split
        pop = self.population if target_pop == "active" else self.all_prompts
        for prompt in prompts:
            self.data.eval_on_batch(prompt, batch)
        if self.data.grading_function is None:
            pop.binary_tournament_sort(batch)
        pop.prompts.sort(key=lambda p: p.get_score(split), reverse=True)
        pop.ranked = True

    def __run(self):
        for step in range(self.start_gen + 1, self.start_gen + ITER + 1):
            print("step", step)
            self.all_prompts.set_update(self.population.prompts)
            self.all_prompts.dump()
            # n = self.population.purge_duplicates()
            n = self.population.purge_worst()
            new_prompts = self.op(gen=step, n=n)
            if self.data.grading_function is None:
                batch = self.start_batch
            else:
                batch = self.data.get_batch("dev", n=BATCH_SIZE)
            self.eval_and_sort(new_prompts, batch)
            self.population.dump(gen=step)
        # add last generation
        self.all_prompts.set_update(self.population.prompts)

    def load_and_fill_population(self, initial_population: list[Prompt] = []):
        if len(initial_population) > 0:
            self.all_prompts.set_update(initial_population)
            active = [p for p in self.all_prompts if p.active]
            self.population.add(active)
            self.start_gen = max([p.gen for p in self.population])

        remaining = POP_SIZE
        if len(initial_population) < POP_SIZE:
            remaining = POP_SIZE - len(initial_population)
        self.lamarckian(n=remaining, gen=self.start_gen)

    def begin(self, initial_population: list[Prompt] = []):
        self.load_and_fill_population(initial_population)

        print(f"Starting eval on batch {self.start_batch}")
        print(f"Population size: {len(self.population.prompts)}")
        self.eval_and_sort(self.population.prompts, self.start_batch)

        self.all_prompts.set_update(self.population.prompts)
        self.population.dump(gen=0)
        self.all_prompts.dump()

        logger.info("Starting optimization")
        self.__run()
        logger.info("Optimization done")
        self.all_prompts.dump()
        self.data.dump()

    def eval(self):
        logger.info("Starting final eval")
        # if self.data.test[0].gold is None:
        if self.data.grading_function is None:
            print("Gold-free test")
            self.eval_and_sort(
                self.all_prompts.prompts,
                self.data.get_batch("test", n=BATCH_SIZE),
                "all",
            )
            by_gen = [
                [prompt.get_score("test") for prompt in gen]
                for gen in self.all_prompts.filter_by_iteration()
                if len(gen) > 0
            ]
        else:
            print("Gold label testing")
            by_gen = self.all_prompts.test_iterations(self.data)
        logger.info("Final eval done")
        self.all_prompts.dump()
        print(by_gen)
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
        self.data.dump()
