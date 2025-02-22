from data import Data
from prompt import Prompt
import json
import os
import random
from typing import Callable
import os
import logging
import Levenshtein

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

        # "tool_name": (uses, average_score)
        self.tool_effectivity = {
            "lamarckian": (0,0.0),
            "reflective": (0,0.0),
            "iterative": (0,0.0),
            "crossover": (0,0.0),
        }

    def update_tool_effectivity(self, tool: str, score: float) -> None:
        count = self.tool_effectivity[tool][0] + 1
        prev_avg = self.tool_effectivity[tool][1] 
        new_avg = prev_avg + (score-prev_avg)/count
        self.tool_effectivity[tool] = (count, new_avg)

    def add(self, prompts: Prompt | list[Prompt]) -> None:
        if type(prompts) == Prompt:
            prompts = [prompts]
        for prompt in prompts:
            self.prompts.append(prompt)
            self.prompts.sort(key=lambda p: p.get_score("dev"), reverse=True)

    def set_update(self, prompts: list[Prompt]) -> None:
        s = set(self.prompts)
        s.update(set(prompts))
        self.prompts = list(s)
        
    def __iter__(self):
        return iter(self.prompts)
    
    def top_n(self, n: int) -> list[Prompt]:
        return self.prompts[:n]

    def select(self, n: int) -> list[Prompt]:
        counts = [p.score_to_count() for p in self.prompts]
        return random.sample(self.prompts, n, counts=counts)

    def stats(self) -> tuple[float, float]:
        scores = [p.get_dev() for p in self.prompts]
        self.avg_score = sum(scores) / len(scores)
        self.max_score = max(scores)
        return self.avg_score, self.max_score

    def dump(self, gen = None):
        fn = "prompts"
        if gen:
            fn += f"{gen}"
        with open(f"{os.getenv('RUN_FOLDER')}/{fn}.jsonl", "w", encoding="utf-8") as f:
            for prompt in self.prompts:
                json.dump(prompt.jsoned(), f)
                f.write("\n")
                
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, index):
        return self.prompts[index]

    def filter_by_iteration(self) -> list[list[Prompt]]:
        max_gen = max([p.gen for p in self.prompts])
        return [list(filter(lambda p: p.gen == i, self.prompts)) for i in range(max_gen+1)]
    
    def evaluate_iterations(self, data: Data, top_n: int = -1) -> list[list[float]]:
        generations = self.filter_by_iteration()
        scores_by_gen = []
        for gen in generations:
            gen_scores = []
            gen = gen if top_n == -1 else gen[:top_n]
            for prompt in gen:
                score = prompt.score(data, self.solve, final=True)
                gen_scores.append(score)
            scores_by_gen.append(gen_scores)
        return scores_by_gen
    
    def quartile(self, i: int) -> list[Prompt]:
        quarter = len(self.prompts) //4
        return self.prompts[(i-1)*quarter:i*quarter]

    ## POPULATION CONTROL TOOLS

    def purge_worst(self) -> int:
        """
        Remove the worst quarter (1/4) of prompts from the population.
        Begins new generation.

        Args:
            None

        Returns:
            int: How many were purged
        """
        self.population.dump()
        purged = len(self)//4
        for i in range(purged):
            logger.info(f"PURGE WORST ({i}): {self.prompts[-1].text}")
            self.population.prompts[-1].active = False
            self.population.prompts.pop()
        self.population.current_gen += 1
        return purged

    def purge_duplicates(self) -> int:
        """
        After sorting by score, go prompt by prompt and remove the most similar prompt until a quarter (1/4) of the population is deleted.
        Begins new generation.

        Args:
            None

        Returns:
            int: How many were purged
        """
        self.population.dump()
        purged = len(self)//4
        for i in range(purged):
            curr = self.population[i]
            most_similar = sorted(self.prompts[i+1:], key=lambda p: Levenshtein.distance(curr.text, p.text))[0]
            most_similar.active = False
            most_similar_ix = self.population.prompts.index(most_similar)
            self.population.prompts.pop(most_similar_ix)
            logger.info(f"PURGE DUPLICATES ({i}): {most_similar.text}")
        self.population.current_gen += 1
        return purged