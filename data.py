import dspy
import json
from typing import Literal, Optional
import os
from prompt import Prompt
import logging
import utils
import random
import grading

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/scores.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



SOLVE_TEMP = 0.0


class Data:
    def __init__(self, data: list[dspy.Example]):
        self.data = data
        self.train, self.dev, self.test = self.__get_splits()
        self.scores = {x: [-1]*len(self.__getattribute__(x)) for x in ["train", "dev", "test"]}
        self.solve_handle = self.__get_solve_handle()

    def length(self, split: Literal["all", "train", "dev", "test"] = "all"):
        return len(self.data) if split == "all" else len(self.__getattribute__(split))
    
    def select(self, split: Literal["train", "dev", "test"], n: int) -> list[dspy.Example]:
        assert split in ["train", "dev", "test"]
        data = self.__getattribute__(split)
        if n < len(data):
            return random.sample(data, n)
        return data
    
    def __get_splits(self):
        ss = len(self.data) // 3
        return self.data[:ss], self.data[ss:2*ss], self.data[2*ss:]

    def __get_solve_handle(self):
        solve_lm = utils.get_lm("SOLVE")
        
        solve_module = dspy.ChainOfThought(grading.str_signature, temperature=SOLVE_TEMP)
            
        def solve(question):
            try:
                with dspy.context(lm=solve_lm):
                    ret = solve_module(question=question)
                return ret
            except ValueError as e:
                return None
            
        return solve
            
    @classmethod
    def from_json(cls, path):
        with open(f"{path}", "r") as f:
            data = json.load(f)
        def examplify(s):
            ret = dspy.Example(question=s["question"], answer=s["answer"])
            ret.with_inputs("question")
            return ret
        
        data = map(examplify, data)
        return cls(list(data))

    def __str__(self):
        TEMPLATE = "Question: {q}\nAnswer: {a}"
        example_strings = [TEMPLATE.format(q=e.question, a=e.answer) for e in self.data]
        return "\n".join(example_strings)

    def eval_on_split(self, 
                      prompt: Prompt, 
                      split: Literal["train", "dev", "test"] = "dev",
                      batch_size: int = 5):
        assert split in ["train", "dev", "test"]
        data: list[dspy.Example] = self.__getattribute__(split) 
        if data == None:
            raise ValueError(f"Wrong split name {split}")
        
        if batch_size > 0 and len(data) > batch_size:
            data = random.sample(data, batch_size)

        def get_reasoning(completions, idx):
            if hasattr(completions, "rationale"):
                return completions.rationale[idx]
            elif hasattr(completions, "reasoning"):
                return completions.reasoning[idx]
            else: 
                return None
            
        old_score = prompt.get_score(split)
        if old_score > -1.0:
            logger.warning(f"Tried grading prompt {prompt.text} with assigned score {old_score}.")
            return old_score
        if not prompt.valid:
            logger.warning(f"Tried grading prompt {prompt.text} which is invalid.")
            prompt.set_score(split, 0.0)
            return 0.0

        batch_score = 0.0
        for i, example in enumerate(data):
            question = prompt.format(example.question)
            response = self.solve_handle(question=question)
            if response:
                solutions = response.completions
                gold = example.answer
                acc_score_on_sample = 0.0
                N_SOLUTIONS = len(solutions.answer)
                logger.debug(f"Grading problem {i+1} on split {split}")
                for comp_idx in range(N_SOLUTIONS):
                    rationale = get_reasoning(solutions, comp_idx)
                    solution = solutions.answer[comp_idx]
                    #grade = grading.score_connections(solution, gold, logger=logger)
                    grade = grading.exact_match_float(solution, gold, logger=logger)
                    acc_score_on_sample += grade
                    logger.debug(f"Completion {comp_idx+1}\nRationale:{rationale}\nSolution:{solution}\t|\tGold:{gold}\nPass:{grade}\n")
                prompt.completions.append((example, get_reasoning(response.completions, 0), grade))
                avg_score_on_sample = acc_score_on_sample / N_SOLUTIONS
                self.scores[split][i] = avg_score_on_sample
                batch_score += avg_score_on_sample
        prompt.set_score(split, batch_score / len(data))
        return prompt.get_score(split)

