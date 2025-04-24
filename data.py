import json
from typing import Literal
import os
from prompt import Prompt
import logging
import random
import grading
from model_api import model
import my_signatures as sig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/scores.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



SOLVE_TEMP = 0.0

class Data:
    def __init__(self, data: list[dict], answer_type: type = str):
        self.data = data
        self.train, self.dev, self.test = self.__get_splits()
        self.scores = {x: [-1]*len(self.__getattribute__(x)) for x in ["train", "dev", "test"]}
        self.answer_type = answer_type
        self.solve_handle = self.__get_solve_handle()

    def length(self, split: Literal["all", "train", "dev", "test"] = "all"):
        return len(self.data) if split == "all" else len(self.__getattribute__(split))
    
    def select(self, split: Literal["train", "dev", "test"], n: int) -> list[dict]:
        assert split in ["train", "dev", "test"]
        data = self.__getattribute__(split)
        if n < len(data):
            return random.sample(data, n)
        return data
    
    def __get_splits(self):
        ss = len(self.data) // 3
        return self.data[:ss], self.data[ss:2*ss], self.data[2*ss:]

    def __get_solve_handle(self):
        solve_sig = sig.Signature.from_str(f"task: str (task to be solved) -> solution: {self.answer_type} ()")
        def solve(question):
            try:
                ret = model.chain_of_thought(solve_sig, temp=SOLVE_TEMP, task=question)
                return ret
            except Exception as e:
                logger.warning(f"Solve exception: {e}")
            
        return solve
            
    @classmethod
    def from_json(cls, path, answer_type):
        with open(f"{path}", "r") as f:
            data = json.load(f)
        return cls(data, answer_type)

    def __str__(self):
        TEMPLATE = "Question: {q}\nAnswer: {a}"
        example_strings = [TEMPLATE.format(q=e["question"], a=e["answer"]) for e in self.data]
        return "\n".join(example_strings)

    def eval_on_split(self, 
                      prompt: Prompt, 
                      split: Literal["train", "dev", "test"] = "dev",
                      batch_size: int = 5):
        assert split in ["train", "dev", "test"], f"Wrong split name {split}"
        data: list[dict] = self.__getattribute__(split) 
        
        if batch_size > 0 and len(data) > batch_size:
            data = random.sample(data, batch_size)  
            
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
            question = prompt.format(example["question"])
            logger.debug(f"Grading problem {i+1} on split {split}")
            response = self.solve_handle(question=question)
            if response:
                gold = example["answer"]
                reasoning, solution_dict = response
                solution = solution_dict["solution"]
                logger.debug(f"Got response with \nreasoning: {reasoning}\nsolution: {solution}")
                grade = grading.exact_match_float(solution, gold, logger=logger)
                prompt.completions.append((example, reasoning, solution, grade))
                self.scores[split][i] = grade
                batch_score += grade
                logger.debug(f"Problem {i+1} score: {grade}")
        prompt.set_score(split, batch_score / len(data))
        return prompt.get_score(split)

