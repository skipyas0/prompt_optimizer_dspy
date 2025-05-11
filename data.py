import json
from typing import Literal
import os
from prompt import Prompt
import logging
import random
import grading
from model_api import solve_model
import my_signatures as sig
import uuid
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/scores.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


SOLVE_TEMP = 0.0


class Attempt:
    def __init__(self, prompt_id, example_id, response, grade, id=None):
        self.prompt_id = prompt_id
        self.example_id = example_id
        self.reasoning, self.answer = response if response is not None else (None, None)
        self.grade = grade
        self.id = uuid.uuid4().hex if id is None else id

    def as_cot(self):
        return {"reasoning": self.reasoning, "answer": self.answer}

    @classmethod
    def from_json(cls, json_dict: dict):
        data = json.loads(json_dict)
        return cls(
            prompt_id=data["prompt_id"],
            example_id=data["example_id"],
            response=(data["reasoning"], data["answer"]),
            grade=data["grade"],
            id=data["id"],
        )


class Example:
    def __init__(self, question, gold=None, id=None):
        self.question = question
        self.gold = gold
        self.id = uuid.uuid4().hex if id is None else id
        self.attempts = []
        self.split = None

    def __repr__(self):
        return self.id

    def qa_dict(self):
        if self.gold is None:
            return {"question": self.question}
        elif isinstance(self.gold, tuple):
            return {"question": self.question, "answer": self.gold[-1]}
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

    @classmethod
    def from_json(cls, data: dict):
        attempts = [Attempt.from_json(a) for a in data["attempts"]]
        example = cls(data["question"], data["gold"], data["id"])
        example.attempts = attempts
        return example


class Data:
    def __init__(
        self,
        data: list[dict] | dict[str, list[Example]],
        grading_function,
        eval_function,
        answer_type: str = "str",
    ):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            if "answer" in data[0]:
                self.data = [
                    Example(d["question"], gold=d["answer"], id=d["id"]) for d in data
                ]
            else:
                self.data = [
                    Example(
                        d["question"],
                        gold=(d["test_inputs"], d["test_outputs"], d["code"]),
                        id=d["id"],
                    )
                    for d in data
                ]
            self.train, self.dev, self.test = self.__get_splits()
        elif isinstance(data, dict):
            self.train = [Example.from_json(e) for e in data["train"]]
            self.dev = [Example.from_json(e) for e in data["dev"]]
            self.test = [Example.from_json(e) for e in data["test"]]
            self.data = self.train + self.dev + self.test
        else:
            raise ValueError(
                "Data must be a list of dictionaries or a tuple of three lists of examples."
            )
        self.answer_type = answer_type
        self.solve_handle = self.__get_solve_handle()
        self.grading_function = grading_function
        self.eval_function = eval_function

    def update_attempts(self, attempts: list[Attempt]):
        for attempt in attempts:
            for example in self.data:
                if example.id == attempt.example_id:
                    example.attempts.append(attempt)
                    break
            else:
                raise ValueError(f"Example with id {attempt.example_id} not found.")
        self.dump()

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
            "metadata": {
                "grading_function": self.grading_function.__name__
                if self.grading_function
                else None,
                "eval_function": self.eval_function.__name__
                if self.eval_function
                else None,
                "answer_type": self.answer_type,
            },
        }

    def dump(self):
        with open(
            f"{os.getenv('RUN_FOLDER')}/task_attempts.json", "w+", encoding="utf-8"
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
        if len(self.data) < 3:
            logger.warning("Not enough data to split into train, dev, and test.")
            print("Not enough data to split into train, dev, and test.")
            return self.data, self.data, self.data
        ss = len(self.data) // 3
        train = self.data[:ss]
        dev = self.data[ss : 2 * ss]
        test = self.data[2 * ss :]
        for name, split in zip(["train", "dev", "test"], [train, dev, test]):
            for example in split:
                example.split = name
        return train, dev, test

    def __get_solve_handle(self):
        if self.answer_type == "imageurl":
            load_dotenv()
            client = OpenAI(api_key=os.getenv("API_KEY_UNIVERSAL"))

            def solve(question):
                try:
                    result = client.images.generate(
                        model="dall-e-3", prompt=question, size="1024x1024"
                    )
                    return result.data[0].url
                except Exception as e:
                    logger.warning(f"Solve exception: {e}")
                    return None
        else:
            solve_sig = sig.Signature.from_str(
                f"task: str (task to be solved) -> solution: {self.answer_type} ()"
            )

            def solve(question):
                try:
                    ret = solve_model.chain_of_thought(
                        solve_sig, temp=SOLVE_TEMP, max_tries=1, task=question
                    )
                    return ret
                except Exception as e:
                    logger.warning(f"Solve exception: {e}")
                    return None
        return solve

    @classmethod
    def from_json(cls, path, grading_function, eval_function, answer_type: type = str):
        with open(f"{path}", "r") as f:
            data = json.load(f)
        return cls(data, grading_function, eval_function, answer_type)

    @classmethod
    def from_attempts_file(cls, path):
        with open(f"{path}", "r") as f:
            data = json.load(f)
        metadata = data["metadata"]
        grading_function = getattr(grading, metadata["grading_function"])
        eval_function = getattr(grading, metadata["eval_function"])
        answer_type = metadata["answer_type"]

        splits = {
            s: [Example.from_json(e) for e in data[s]] for s in ["train", "dev", "test"]
        }

        return cls(splits, grading_function, eval_function, answer_type)

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

    def eval_on_batch(self, prompt: Prompt, batch: list[Example], phase="optim"):
        if not prompt.valid:
            logger.warning(f"Tried grading prompt {prompt.text} which is invalid.")
            prompt.set_score("dev", 0.0)
            prompt.set_score("test", 0.0)
            return 0.0
        grading_function = (
            self.grading_function if phase == "optim" else self.eval_function
        )
        if (
            grading_function is not None
            and prompt.get_score("dev") > -1.0
            and phase == "optim"
        ):
            logger.warning(
                f"Tried grading prompt {prompt.text} which is already graded."
            )
            return prompt.get_score("dev")

        batch_score = None if grading_function is None else 0.0
        split = "dev" if self.answer_type=="imageurl" else batch[0].split # workaround for imageurl 
        for i, example in enumerate(batch):
            llm_input = prompt.format(example.question)
            logger.debug(
                f"Grading problem id {example.id} on split {split} using prompt {prompt.id}"
            )
            response = self.solve_handle(question=llm_input)
            grade = None
            if response:
                gold = example.gold
                if self.answer_type == "imageurl":
                    solution = response[:]
                    reasoning = None
                    response = (None, solution)
                else:
                    reasoning, solution_dict = response
                    solution = None if solution_dict is None else solution_dict["solution"]
                logger.debug(
                    f"Got response with \nreasoning: {reasoning}\nsolution: {solution}"
                )
                if batch_score is not None:
                    grade = grading_function(solution, gold)
                    batch_score += grade
            logger.debug(f"Problem {i + 1} score: {grade}")
            attempt = Attempt(prompt.id, example.id, response, grade)
            example.attempts.append(attempt)
            prompt.attempts.append(attempt)
            print(
                f"Added attempt {attempt.id} to example {example.id} and prompt {prompt.id}"
            )
            print(
                f"Example {example.id} has {len(example.attempts)} attempts and prompt {prompt.id} has {len(prompt.attempts)} attempts"
            )
            self.dump()
        if batch_score is not None:
            prompt.set_score(split, batch_score / len(batch))
        return prompt.get_score(split)
