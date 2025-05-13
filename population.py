from data import Data, Example
from prompt import Prompt
from typing import Optional
import json
import os
import random
import logging
import Levenshtein
import my_signatures as sig
from model_api import optim_model
import math
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Population:
    def __init__(self, prompts: list[Prompt]) -> None:
        """
        Structure that holds prompts during optimization.
        """
        self.prompts: list[Prompt] = prompts
        self.avg_score, self.max_score = -1.0, -1.0
        self.comparisons = []

    def get_by_id(self, id: str) -> Optional[Prompt]:
        """
        Get prompt by id.
        """
        for prompt in self.prompts:
            if prompt.id == id:
                return prompt
        return None

    def add(self, prompts: Prompt | list[Prompt]) -> None:
        if isinstance(prompts, Prompt):
            prompts = [prompts]
        for prompt in prompts:
            self.prompts.append(prompt)

    def binary_tournament_sort(self, batch: list[Example]) -> None:
        """
        Compare every prompt with every other prompt using LLM-as-a-judge on outputs.
        """
        split = batch[0].split
        for task in batch:
            for i in range(len(self.prompts)):
                for j in range(i + 1, len(self.prompts)):
                    prompts = [self.prompts[i], self.prompts[j]]
                    random.shuffle(prompts)
                    prompt_a, prompt_b = prompts[0], prompts[1]
                    self.compare_prompts(prompt_a, prompt_b, task, split)

    def compare_prompts(self, prompt_a: Prompt, prompt_b: Prompt, task: Example, split: str) -> None:
        """
        Uses LLM-as-a-judge to compare two prompts based on performance on the same task.
        """

        logger.debug(f"Comparing {prompt_a.id} and {prompt_b.id} on task {task.id}")

        # get attempts on the same task from both prompts
        try:
            attempt_a = [a for a in prompt_a.attempts if a.example_id == task.id][-1]
            attempt_b = [a for a in prompt_b.attempts if a.example_id == task.id][-1]
        except Exception as e:
            logger.warning(f"Comparison failed, no attempts available: {e}")
            return

        if split is None: # this is an image generation task, needs different cot call
            split = "dev"
            _, comparison = optim_model.chain_of_thought(
                sig.image_compare,
                description=task.question,
                prompt_a=prompt_a.text,
                prompt_b=prompt_b.text,
                output_a=attempt_a.answer,
                output_b=attempt_b.answer,
            )
        else: # all other tasks
            _, comparison = optim_model.chain_of_thought(
                sig.compare,
                task_question=task.qa_dict(),
                prompt_a=prompt_a.text,
                prompt_b=prompt_b.text,
                output_a=attempt_a.as_cot(),
                output_b=attempt_b.as_cot(),
            )

        output_comp = comparison["output_comparison"]
        prompt_comp = comparison["prompt_comparison"]
        verdict = comparison["verdict"]

        # in debug mode we dont call llms -> random choice of winner
        if os.getenv("DEBUG") is not None:
            verdict = random.choice(["prompt_a", "prompt_b"])
            logger.debug(f"DEBUG: {verdict} chosen")

        # save comparison 
        comparison_log = {
            "attempt": attempt_a.id,
            "split": split,
            "output_comp": output_comp,
            "prompt_comp": prompt_comp,
            "prompt_a": prompt_a.id,
            "prompt_b": prompt_b.id,
            "verdict": verdict,
        }
        prompt_a.update_comparisons(comparison_log) # update win rates
        prompt_b.update_comparisons(comparison_log) # update win rates
        self.comparisons.append(comparison_log)

        
        if verdict == "prompt_a":
            attempt_a.grade = 1
            attempt_b.grade = 0
        elif verdict == "prompt_b":
            attempt_a.grade = 0
            attempt_b.grade = 1
        else:
            # if llm failed to select one, assume draw
            logger.warning(
                f"Invalid verdict {verdict} for prompts {prompt_a.id} and {prompt_b.id}"
            )
            attempt_a.grade = 0.5
            attempt_b.grade = 0.5

    def set_update(self, prompts: list[Prompt]) -> None:
        """
        Add prompts while avoiding duplicating.
        """
        s = set(self.prompts)
        s.update(set(prompts))
        self.prompts = list(s)

    def __iter__(self):
        return iter(self.prompts)

    def select(self, n: int) -> list[Prompt]:
        """
        Select n prompts from population with probability proportional to their scores.
        """
        if n >= len(self.prompts):
            return self.prompts
        scores = [p.get_score("dev") for p in self.prompts]

        # linear piecewise fun to generate roulette sampling counts
        def softmax_like(score):
            if score < 0.0:
                return 0
            elif score == 0.0:
                return 1
            elif score >= 1.0:
                return 10
            else:
                return 1 + (score * 9)

        counts = [math.floor(softmax_like(score)) for score in scores]
        return random.sample(self.prompts, n, counts=counts)

    def stats(self) -> tuple[float, float]:
        scores = [p.get_dev() for p in self.prompts]
        self.avg_score = sum(scores) / len(scores)
        self.max_score = max(scores)
        return self.avg_score, self.max_score

    def dump(self, gen: Optional[int] =None) -> None:
        """
        Save all prompt data and comparisons to json.
        """

        fn = "prompts"
        if gen is not None:
            fn += f"{gen}"
        with open(f"{os.getenv('RUN_FOLDER')}/{fn}.jsonl", "w", encoding="utf-8") as f:
            for prompt in self.prompts:
                json.dump(prompt.to_dict(), f)
                f.write("\n")
        if len(self.comparisons) > 0:
            with open(
                f"{os.getenv('RUN_FOLDER')}/comparisons.json", "w", encoding="utf-8"
            ) as f:
                json.dump(self.comparisons, f)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def filter_by_iteration(self) -> list[list[Prompt]]:
        max_gen = max([p.gen for p in self.prompts])
        return [
            list(filter(lambda p: p.gen == i, self.prompts)) for i in range(max_gen + 1)
        ]

    def test_iterations(self, data: Data, phase: str ="optim") -> list[list[float]]:
        """
        Filters prompts by generation and evaluates them.
        """
        scores_by_gen = [
            [data.eval_on_batch(prompt, data.test, phase=phase) for prompt in gen]
            for gen in self.filter_by_iteration()
            if len(gen) > 0
        ]
        return scores_by_gen

    ## POPULATION CONTROL TOOLS

    def purge_worst(self) -> int:
        """
        Remove the worse half of prompts from the population.
        """
        self.dump()
        purged = max(min(len(self) // 2, 10), 1)
        for i in range(purged):
            logger.info(f"PURGE WORST ({i}): {self.prompts[-1].id}")
            self.prompts[-1].active = False
            self.prompts.pop()
        return purged

    def purge_duplicates(self) -> int:
        """
        After sorting by score, go prompt by prompt and remove the most similar prompt until a half of the population is deleted.
        """
        self.dump()

        n = len(self.prompts)
        if n <= 1:
            self.prompts.clear()
            return 1

        purged = max(min(len(self) // 2, 10), 1)  # clip(pop//4, 1, 10)

        # precalculate similarity matrix
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r = Levenshtein.ratio(self.prompts[i].text, self.prompts[j].text)
                sim_matrix[i, j] = r
                sim_matrix[j, i] = r

        to_purge = set()
        used = set()

        for _ in range(purged):
            # find most similar remaining pair
            best = (-1, -1, -1)  # (sim, i, j)
            for i in range(n):
                if i in used:
                    continue
                for j in range(i + 1, n):
                    if j in used:
                        continue
                    sim = sim_matrix[i, j]
                    if sim > best[0]:
                        best = (sim, i, j)
            _, i, j = best
            # mark one of the pair for purging (e.g. j)
            to_purge.add(j)
            used.add(j)
        for i in sorted(to_purge, reverse=True):
            p = self.prompts.pop(i)
            logger.info(f"PURGE DUPLICATES ({i}): {p.id}")
        return purged
