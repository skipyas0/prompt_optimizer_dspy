import dspy
from prompt import Prompt
import json
import os

class Population:
    def __init__(self, prompts: list[Prompt], solve) -> None:
        self.prompts: list[Prompt] = prompts
        self.avg_score, self.max_score = -1.0, -1.0
        self.ranked = False
        self.solve = solve

    def add(self, prompts: list[Prompt]) -> None:
        self.prompts += prompts
        self.ranked = False

    def sorted(self, data: list[dspy.Example]) -> list[tuple[str, float]]:
        self.prompts.sort(key=lambda p: p.score(data, self.solve), reverse=True)
        scores = [p.score(data, self.solve) for p in self.prompts]
        self.avg_score = sum(scores) / len(scores)
        self.max_score = max(scores)
        self.ranked = True
        return self.prompts
    
    def __iter__(self):
        return iter(self.prompts)
    
    def top_n(self, n: int, data: list[dspy.Example]) -> list[Prompt]:
        return self.prompts[:n] if self.ranked else self.sorted(data)[:n]

    @property
    def stats(self) -> tuple[float, float]:
        return self.avg_score, self.max_score

    def dump(self):
        with open(f"{os.getenv('RUN_FOLDER')}/prompts.jsonl", "w", encoding="utf-8") as f:
            for prompt in self.prompts:
                json.dump(prompt.jsoned(), f)
                f.write("\n")


    def filter_by_iteration(self) -> list[list[Prompt]]:
        max_gen = max([p.gen for p in self.prompts])
        return [list(filter(lambda p: p.gen == i, self.prompts)) for i in range(max_gen+1)]
    
    def evaluate_iterations(self, data: list[dspy.Example], top_n: int = -1) -> list[list[float]]:
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