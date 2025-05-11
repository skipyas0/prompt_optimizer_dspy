import logging
import os
from prompt import Prompt
from population import Population
import matplotlib.pyplot as plt
import random
from data import Data, Example
import my_signatures as sig
from model_api import optim_model, solve_model
import json


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Optimizer:
    def __init__(self, data: Data, settings: dict, initial_comparisons=[]):
        
        self.start_gen = 0
        self.population = Population([])
        self.population.comparisons = initial_comparisons
        self.all_prompts = Population([])
        self.all_prompts.comparisons = initial_comparisons

        self.operators = {
            "LAMARCKIAN": self.image_lamarckian if data.answer_type == "imageurl" else self.lamarckian,
            "REFLECTIVE": self.reflective,
            "ITERATIVE": self.iterative,
            "PARAPHRASE": self.paraphrase,
            "FEEDBACK": self.feedback,
        }
        self.settings = settings
        self.op = self.operators[self.settings["operator"]]
        with open("datasets/seeds/values.json", "r") as f:
            user_values = json.load(f)
        self.random_value_focus = lambda: random.sample(user_values, 3)

        with open("datasets/seeds/personas.json", "r") as f:
            personas = json.load(f)
        self.random_persona = lambda: random.choice(personas)

        logger.info(f"Settings: {self.settings["pop_size"]=}, {self.settings["max_iters"]=}, {self.settings["batch_size"]=}")
        self.data = data
        logger.info(
            f"Split lengths: train {len(self.data.train)}, dev {len(self.data.dev)}, test {len(self.data.test)}"
        )
        self.start_batch = self.data.get_batch("dev", n=self.settings["batch_size"])
        print(self.data.data, self.data.train, self.data.dev, self.data.test)
        print(self.start_batch)
        
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

        prompts = []
        seeding_source = self.settings["seeding_source"]
        origin = "lamarckian_no_seed" if seeding_source == "NOSEED" else "lamarckian_personas"
        for _ in range(n):
            examples = [e.qa_dict() for e in self.data.select("train", self.settings["lamarck_batch"])]
            random.shuffle(examples)
            if seeding_source == "NOSEED":
                _, completion = optim_model.chain_of_thought(
                    sig.lamarckian, task_examples=examples
                )
            elif seeding_source == "PERSONAS":
                _, completion = optim_model.chain_of_thought(
                    sig.lamarckian_personas, task_examples=examples, persona=self.random_persona()
                )
            else:
                raise ValueError(f"Invalid seed source {seeding_source}")


            prompt_obj = Prompt(
                completion["prompt_proposal"], origin=origin, gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(f"LAMARCKIAN, seed {seeding_source}, generated prompt:\n {str(prompt_obj)}\n")
            prompts.append(prompt_obj)
        return prompts
    
    def image_lamarckian(self, gen: int = 0, n: int = 1) -> list[Prompt]:
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

        prompts = []
        origin = "image_lamarckian"
        for _ in range(n):
            examples = [e.qa_dict() for e in self.data.data]
            _, completion = optim_model.chain_of_thought(
                sig.image_lamarckian, description=examples[0]["question"]
            )

            prompt_obj = Prompt(
                completion["prompt_proposal"], origin=origin, gen=gen, placeholder=None
            )
            self.population.add(prompt_obj)
            logger.info(f"IMAGE LAMARCKIAN, generated prompt:\n {str(prompt_obj)}\n")
            prompts.append(prompt_obj)
        return prompts
    
    def reflective(self, gen: int = 0, n: int = 1) -> list[Prompt]:
        """ """
        prompts = []
        prompts_with_attempts = list(filter(lambda p: len(p.attempts)>0, self.population.prompts))
        for prompt in prompts_with_attempts:
            print(f"Prompt {prompt.id} has {len(prompt.attempts)} attempts")
        if len(prompts_with_attempts) == 0:
            logger.warning("No prompts with attempts, using lamarckian insted of reflective")
            return self.lamarckian(gen=gen, n=n)
        
        
        prompts_with_attempts = sorted(
            prompts_with_attempts, key=lambda p: min([a.grade for a in p.attempts])
        )
        for _ in range(n):
            original = prompts_with_attempts[0]
            solution = sorted(original.attempts, key=lambda a: a.grade)[-1]

            example_id = solution.example_id
            example = self.data.get_by_id(example_id)
            reasoning = solution.reasoning
            _, completion = optim_model.chain_of_thought(
                signature=sig.reflective,
                original_prompt=original.text,
                task_question=example.question,
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
            signature = sig.image_feedback if self.data.answer_type == "imageurl" else sig.feedback
            _, completion = optim_model.chain_of_thought(
                signature=signature,
                base_prompt=original.text,
                comparisons=comparisons,
            )

            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="feedback", gen=gen
            )
            if self.data.answer_type == "imageurl":
                prompt_obj.placeholder = None
            self.population.add(prompt_obj)
            logger.info(
                f"FEEDBACK generated prompt:\n {str(prompt_obj)}\nOriginal prompt:\n{original.text}\n"
            )
            prompts.append(prompt_obj)
        return prompts

    def iterative(self, gen: int = 0, n: int = 1) -> list[Prompt]:
        """ """
        prompts = []
        for _ in range(n):
            examples = [p.prompt_and_perf() for p in self.population.select(5)]
            examples = sorted(examples, key=lambda t: t[1])
            _, completion = optim_model.chain_of_thought(sig.iterative, old_prompts=examples)
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="iterative", gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(
                f"ITERATIVE generated prompt:\n {str(prompt_obj)}\nExamples:\n{examples}\n"
            )
            prompts.append(prompt_obj)
        return prompts

    def paraphrase(self, gen: int = 0, n: int = 1):
        """ """
        prompts = []
        for _ in range(n):
            input_prompt = self.population.select(1)[0]
            _, completion = optim_model.chain_of_thought(
                sig.paraphrase, original_prompt=input_prompt.text
            )
            prompt_obj = Prompt(
                completion["prompt_proposal"], origin="paraphrase", gen=gen
            )
            self.population.add(prompt_obj)
            logger.info(
                f"PARAPHRASE generated prompt:\n {str(prompt_obj)}\n from prompt:{input_prompt}\n."
            )
            prompts.append(prompt_obj)
        return prompts

    def eval_and_sort(
        self, prompts: list[Prompt], batch: list[Example], target_pop="active"
    ):
        split = "dev" if self.data.answer_type=="imageurl" else batch[0].split
        pop = self.population if target_pop == "active" else self.all_prompts
        for prompt in prompts:
            self.data.eval_on_batch(prompt, batch)
        if self.data.grading_function is None:
            pop.binary_tournament_sort(batch)
        pop.prompts.sort(key=lambda p: p.get_score(split), reverse=True)
        pop.ranked = True

    def __run(self):
        for step in range(self.start_gen + 1, self.start_gen + self.settings["max_iters"] + 1):
            print("step", step)
            self.all_prompts.set_update(self.population.prompts)
            self.all_prompts.dump()
            # n = self.population.purge_duplicates()
            print(f"Population size: {len(self.population.prompts)}")
            if self.settings["purge"] == "duplicates":
                n = self.population.purge_duplicates()
            elif self.settings["purge"] == "worst":
                n = self.population.purge_worst()
            print(f"New pop size: {len(self.population.prompts)}, purged {n}")
            new_prompts = self.op(gen=step, n=n)
            print(f"New prompts: {len(new_prompts)}")
            if self.data.grading_function is None:
                batch = self.start_batch
            else:
                batch = self.data.get_batch("dev", n=self.settings["batch_size"])
            self.eval_and_sort(new_prompts, batch)
            self.population.dump(gen=step)
        # add last generation
        self.all_prompts.set_update(self.population.prompts)

    def load_and_fill_population(self, initial_population: list[Prompt] = []):
        if len(initial_population) > 0:
            if self.data.answer_type == "imageurl":
                for p in initial_population:
                    p.placeholder = None
            self.all_prompts.set_update(initial_population)
            active = [p for p in self.all_prompts if p.active]
            self.population.add(active)
            self.start_gen = max([p.gen for p in self.population])

        remaining = 0
        if len(initial_population) < self.settings["pop_size"]:
            remaining = self.settings["pop_size"] - len(initial_population)
        print(f"Filling population with {remaining} prompts")

        self.operators["LAMARCKIAN"](n=remaining, gen=self.start_gen)

    def begin(self, initial_population: list[Prompt] = []):
        self.load_and_fill_population(initial_population)
        
        self.all_prompts.set_update(self.population.prompts)
        self.all_prompts.dump()

        print(f"Starting eval on batch {self.start_batch}")
        print(f"Population size: {len(self.population.prompts)}")
        self.eval_and_sort(self.population.prompts, self.start_batch)
        optim_model.token_checkpoint("init")
        solve_model.token_checkpoint("init")
        self.population.dump(gen=0)
        if self.settings["do_optim"]:
            logger.info("Starting optimization")
            self.__run()
            logger.info("Optimization done")
            optim_model.token_checkpoint("optim")
            solve_model.token_checkpoint("optim")
        self.all_prompts.dump()
        self.data.dump()
        if self.settings["do_eval"]:
            self.eval()
            optim_model.token_checkpoint("eval")
            solve_model.token_checkpoint("eval")

    def eval(self):
        logger.info("Starting final eval")
        if self.data.eval_function is not None:
            print("Gold label testing")
            by_gen = self.all_prompts.test_iterations(self.data, "eval")
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
        else:
            logger.warning("No grading function set, skipping final eval")
