import json
from typing import Literal
import os
from prompt import Prompt
import logging
import random
import grading
from model_api import model
import my_signatures as sig
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/scores.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


SOLVE_TEMP = 0.0


class Attempt:
    def __init__(self, prompt_id, example_id, response, grade):
        self.prompt_id = prompt_id
        self.example_id = example_id
        self.reasoning, self.answer = response
        self.grade = grade
        self.id = uuid.uuid4().hex

    def as_cot(self):
        return {"reasoning": self.reasoning, "answer": self.answer}

class Example:
    def __init__(self, question, gold=None):
        self.question = question
        self.gold = gold
        self.id = uuid.uuid4().hex
        self.attempts = []
        self.split = None

    def __repr__(self):
        return self.id
    
    def qa_dict(self):
        if self.gold is None:
            return {"question": self.question}
        return {"question": self.question, "answer": self.gold} 

    def to_dict(self):
        return {
            "question": self.question,
            "gold": self.gold,
            "id": self.id,
            "attempts": [a.__dict__ for a in self.attempts],
        }

    def get_prompt_attempts(self, prompt_id, wanted_grade=None):
        prompt_attempts = [a for a in self.attempts if a.prompt_id == prompt_id]
        if wanted_grade is not None:
            prompt_attempts = [a for a in prompt_attempts if a.grade == wanted_grade]
        return prompt_attempts


class Data:
    def __init__(self, data: list[dict], answer_type: type = str):
        self.data = [Example(d["question"], gold=d["answer"]) for d in data]
        self.train, self.dev, self.test = self.__get_splits()
        # self.scores = {x: [-1]*len(self.__getattribute__(x)) for x in ["train", "dev", "test"]}
        self.answer_type = answer_type
        self.solve_handle = self.__get_solve_handle()
        self.grading_function = os.getenv("GRADING_FUNCTION")
        if self.grading_function == "exact_match":
            self.grading_function = grading.exact_match_float
        elif self.grading_function == "match_lists":
            self.grading_function = grading.match_lists
        elif self.grading_function == "compare":
            self.grading_function = None
        else:
            raise ValueError(f"Unknown grading function {self.grading_function}")
        
    def get_by_id(self, example_id):    
        for example in self.data:
            if example.id == example_id:
                return example
        raise ValueError(f"Example with id {example_id} not found.")

    def get_all_prompt_attempts(
        self, prompt_id, split="train", wanted_grade=None, example_id=None
    ):
        assert split in ["train", "dev", "test"]
        data = (
            self.__getattribute__(split)
            if example_id is None
            else self.get_all_example_attempts(example_id, split)
        )
        prompt_attempts = []
        for example in data:
            prompt_attempts += example.get_prompt_attempts(prompt_id, wanted_grade)
        return prompt_attempts

    def get_all_example_attempts(self, example_id, split="train"):
        assert split in ["train", "dev", "test"]
        data = self.__getattribute__(split)
        return [ex.attempts for ex in data if ex.id == example_id]

    def to_dict(self):
        return {
            "train": [e.to_dict() for e in self.train],
            "dev": [e.to_dict() for e in self.dev],
            "test": [e.to_dict() for e in self.test],
        }

    def dump(self):
        with open(
            f"{os.getenv('RUN_FOLDER')}/task_attempts.jsonl", "w+", encoding="utf-8"
        ) as f:
            json.dump(self.to_dict(), f)

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
        train = self.data[:ss]
        dev = self.data[ss : 2 * ss]
        test = self.data[2 * ss :]
        for name, split in zip(["train", "dev", "test"], [train, dev, test]):
            for example in split:
                example.split = name
        return train, dev, test

    def __get_solve_handle(self):
        solve_sig = sig.Signature.from_str(
            f"task: str (task to be solved) -> solution: {self.answer_type} ()"
        )

        def solve(question):
            try:
                ret = model.chain_of_thought(solve_sig, temp=SOLVE_TEMP, task=question)
                return ret
            except Exception as e:
                logger.warning(f"Solve exception: {e}")
                return None

        return solve

    @classmethod
    def from_json(cls, path, answer_type):
        with open(f"{path}", "r") as f:
            data = json.load(f)
        return cls(data, answer_type)

    def __str__(self):
        TEMPLATE = "Question: {q}\nAnswer: {a}"
        example_strings = [
            TEMPLATE.format(q=e["question"], a=e["answer"]) for e in self.data
        ]
        return "\n".join(example_strings)

    def get_batch(self, split: Literal["train", "dev", "test"], n: int):
        assert split in ["train", "dev", "test"]
        data = self.__getattribute__(split)
        if n < len(data):
            return random.sample(data, n)
        return data
    
    def eval_on_batch(
        self,
        prompt: Prompt,
        batch: list[Example]
    ):
        
        if not prompt.valid:
            logger.warning(f"Tried grading prompt {prompt.text} which is invalid.")
            prompt.set_score("dev", 0.0)
            prompt.set_score("test", 0.0)
            return 0.0

        batch_score = None if self.grading_function is None else 0.0
        split = batch[0].split
        for i, example in enumerate(batch):
            llm_input = prompt.format(example.question)
            logger.debug(f"Grading problem id {example.id} on split {split} using prompt {prompt.id}")
            response = self.solve_handle(question=llm_input)
            grade = None
            if response:
                gold = example.gold
                reasoning, solution_dict = response
                solution = solution_dict["solution"]
                logger.debug(
                    f"Got response with \nreasoning: {reasoning}\nsolution: {solution}"
                )
                if batch_score is not None:
                    grade = self.grading_function(solution, gold)
                    batch_score += grade
            logger.debug(f"Problem {i + 1} score: {grade}")
            attempt = Attempt(prompt.id, example.id, response, grade)
            example.attempts.append(attempt)
            prompt.attempts.append(attempt)

        if batch_score is not None:
            prompt.set_score(
                split, batch_score / len(batch)
            )
        return prompt.get_score(split)
