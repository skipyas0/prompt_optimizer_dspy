from data import Data
from prompt import Prompt
import json
import os
import random
import logging
import Levenshtein
import my_signatures as sig
from model_api import model
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Population:
    def __init__(self, prompts: list[Prompt]) -> None:
        self.prompts: list[Prompt] = prompts
        self.avg_score, self.max_score = -1.0, -1.0
        self.ranked = False
        self.comparisons = []

    def get_by_id(self, id: str) -> Prompt | None:
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
            # self.prompts.sort(key=lambda p: p.get_score("dev"), reverse=True)

    # def normalize_scores(self, split) -> None:
    #     """
    #     Normalize scores to be between 0 and 1.
    #     """
    #     max_score = max([p.get_score(split) for p in self.prompts])
    #     min_score = min([p.get_score(split) for p in self.prompts])
    #     if max_score == min_score:
    #         return
    #     for prompt in self.prompts:
    #         prompt.set_score(
    #             split, (prompt.get_score(split) - min_score) / (max_score - min_score)
    #         )

    def binary_tournament_sort(self, batch) -> None:
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
        # self.normalize_scores(split)

    def compare_prompts(self, prompt_a, prompt_b, task, split):
        logger.debug(f"Comparing {prompt_a.id} and {prompt_b.id} on task {task.id}")
        try:
            attempt_a = [a for a in prompt_a.attempts if a.example_id == task.id][-1]
            attempt_b = [a for a in prompt_b.attempts if a.example_id == task.id][-1]
        except Exception as e:
            logger.warning(f"Comparison failed, no attempts available: {e}")

        _, comparison = model.chain_of_thought(
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

        if os.getenv("DEBUG") is not None:
            verdict = random.choice(["prompt_a", "prompt_b"])
            logger.debug(f"DEBUG: {verdict} chosen")
        comparison_log = {
            "attempt": attempt_a.id,
            "split": split,
            "output_comp": output_comp,
            "prompt_comp": prompt_comp,
            "prompt_a": prompt_a.id,
            "prompt_b": prompt_b.id,
            "verdict": verdict,
        }
        prompt_a.update_comparisons(comparison_log)
        prompt_a.update_comparisons(comparison_log)
        self.comparisons.append(comparison_log)
        if verdict == "prompt_a":
            # prompt_a.set_score(split, prompt_a.get_score(split) + 1)
            attempt_a.grade = 1
            attempt_b.grade = 0
        elif verdict == "prompt_b":
            # prompt_b.set_score(split, prompt_b.get_score(split) + 1)
            attempt_a.grade = 0
            attempt_b.grade = 1
        else:
            logger.warning(
                f"Invalid verdict {verdict} for prompts {prompt_a.id} and {prompt_b.id}"
            )
            prompt_a.set_score(split, prompt_a.get_score(split) + 0.5)
            prompt_b.set_score(split, prompt_b.get_score(split) + 0.5)
            attempt_a.grade = 0.5
            attempt_b.grade = 0.5

    def set_update(self, prompts: list[Prompt]) -> None:
        s = set(self.prompts)
        s.update(set(prompts))
        self.prompts = list(s)

    def __iter__(self):
        return iter(self.prompts)

    def top_n(self, n: int) -> list[Prompt]:
        return self.prompts[:n]

    def select(self, n: int) -> list[Prompt]:
        if n >= len(self.prompts):
            return self.prompts
        scores = [p.get_score("dev") for p in self.prompts]

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
        print(f"Selecting {n} prompts with counts {counts} from {len(self.prompts)}")
        return random.sample(self.prompts, n, counts=counts)

    def stats(self) -> tuple[float, float]:
        scores = [p.get_dev() for p in self.prompts]
        self.avg_score = sum(scores) / len(scores)
        self.max_score = max(scores)
        return self.avg_score, self.max_score

    def dump(self, gen=None):
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

    def test_iterations(self, data: Data, phase="optim") -> list[list[float]]:
        scores_by_gen = [
            [data.eval_on_batch(prompt, data.test, phase=phase) for prompt in gen]
            for gen in self.filter_by_iteration()
            if len(gen) > 0
        ]
        return scores_by_gen

    def quartile(self, i: int) -> list[Prompt]:
        quarter = len(self.prompts) // 4
        return self.prompts[(i - 1) * quarter : i * quarter]

    ## POPULATION CONTROL TOOLS

    def purge_worst(self) -> int:
        """
        Remove the worse half of prompts from the population.
        Begins new generation.

        Args:
            None

        Returns:
            int: How many were purged
        """
        self.dump()
        purged = max(min(len(self) // 2, 10), 1)
        for i in range(purged):
            logger.info(f"PURGE WORST ({i}): {self.prompts[-1].text}")
            self.prompts[-1].active = False
            self.prompts.pop()
        return purged

    def purge_duplicates(self) -> int:
        """
        After sorting by score, go prompt by prompt and remove the most similar prompt until a half of the population is deleted.
        Begins new generation.

        Args:
            None

        Returns:
            int: How many were purged
        """
        self.dump()

        if len(self) == 1:
            self.prompts.pop()
            return 1

        purged = max(min(len(self) // 2, 10), 1)  # clip(pop//4, 1, 10)
        for i in range(purged):
            curr = self[i]
            most_similar = sorted(
                self.prompts[i + 1 :],
                key=lambda p: Levenshtein.distance(curr.text, p.text),
            )[0]
            most_similar.active = False
            self.prompts.remove(most_similar)
            logger.info(f"PURGE DUPLICATES ({i}): {most_similar.text}")
        return purged
