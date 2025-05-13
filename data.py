import json
from typing import Literal, Optional, Callable, Any
import os
from prompt import Prompt
import logging
import random
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
        """
        Data structure that holds individual solution attempts by the LLM.
        """
        self.prompt_id = prompt_id
        self.example_id = example_id
        self.reasoning, self.answer = response if response is not None else (None, None)
        self.grade = grade
        # don't overwrite id when loading from JSON
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
        """
        Data structure representing a single sample from a dataset.
        """
        self.question = question
        self.gold = gold
        self.id = uuid.uuid4().hex if id is None else id
        # Attempt instances relating to this example
        self.attempts = []
        # Split is assigned after split creation
        self.split = None

    def __repr__(self):
        return self.id

    def qa_dict(self):
        if self.gold is None:
            # gold-label free context
            return {"question": self.question}
        elif isinstance(self.gold, tuple):
            # This is triggered on codecontest dataset, where gold[-1] is a solution code
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
        grading_function: Optional[Callable[[Any, Any], float]],
        eval_function: Optional[Callable[[Any, Any], float]],
        answer_type: str = "str",
    ):
        """
        Dataset representation with the following functionality:
         - data loading
         - split generation
         - solution generation
         - solution evaluation
         - attempt tracking
        """
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # loading from a dataset which has not been used yet - no splits
            if "answer" in data[0]:
                self.data = [
                    Example(d["question"], gold=d["answer"], id=d["id"]) for d in data
                ]
            else:
                # special case for codecontests dataset
                self.data = [
                    Example(
                        d["question"],
                        gold=(d["test_inputs"], d["test_outputs"], d["code"]),
                        id=d["id"],
                    )
                    for d in data
                ]
            self.train, self.dev, self.test = self.get_splits()
        elif isinstance(data, dict):
            # continuing another run, data has splits
            self.train = [Example.from_json(e) for e in data["train"]]
            self.dev = [Example.from_json(e) for e in data["dev"]]
            self.test = [Example.from_json(e) for e in data["test"]]
            self.data = self.train + self.dev + self.test
        else:
            raise ValueError(
                "Data must be a list of dictionaries or a dict with split lists."
            )
        
        self.answer_type = answer_type
        self.solve_handle = self.get_solve_handle()
        self.grading_function = grading_function
        self.eval_function = eval_function

    def update_attempts(self, attempts: list[Attempt]):
        """
        Find corresponding example to each attempt and add a reference.
        Only used when loading attempts.
        """
        for attempt in attempts:
            for example in self.data:
                if example.id == attempt.example_id:
                    example.attempts.append(attempt)
                    break
            else:
                raise ValueError(f"Example with id {attempt.example_id} not found.")
        # update json
        self.dump()

    def get_by_id(self, example_id):
        for example in self.data:
            if example.id == example_id:
                return example
        raise ValueError(f"Example with id {example_id} not found.")

    def get_all_prompt_attempts(
        self, prompt_id, split="train", wanted_grade=None, example_id=None
    ):
        """
        Collects all Attempts that fit the requirements. 
        """
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

    def to_dict(self) -> dict:
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

    def dump(self) -> None:
        with open(
            f"{os.getenv('RUN_FOLDER')}/task_attempts.json", "w+", encoding="utf-8"
        ) as f:
            json.dump(self.to_dict(), f)

    

    def get_splits(self) -> tuple[list[Example],list[Example],list[Example]]:
        """
        Creates 3 equal splits: train, dev and test
        """
        if len(self.data) < 3:
            logger.warning("Not enough data to split into train, dev, and test.")
            print("Not enough data to split into train, dev, and test.")
            return self.data, self.data, self.data
        ss = len(self.data) // 3
        train = self.data[:ss]
        dev = self.data[ss : 2 * ss]
        test = self.data[2 * ss :]

        # Assign split to examples
        for name, split in zip(["train", "dev", "test"], [train, dev, test]):
            for example in split:
                example.split = name
        return train, dev, test

    def get_solve_handle(self) -> Callable[[str], str | tuple[str, dict]]:
        """
        Creates task-specific solve handle for prompt testing and evaluation.
        Try-except wrapper.
        """

        if self.answer_type == "imageurl":
            load_dotenv()
            client = OpenAI(api_key=os.getenv("API_KEY_UNIVERSAL"))

            # For image generation tasks, use dalle
            def solve(question):
                try:
                    result = client.images.generate(
                        model="dall-e-3", prompt=question, size="1024x1024"
                    )
                    # return tuple to make it compatible with structures expecting cot
                    return None, result.data[0].url
                except Exception as e:
                    logger.warning(f"Solve exception: {e}")
                    return None
        else:
            # Signature for specific answer type
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

    def __str__(self):
        TEMPLATE = "Question: {q}\nAnswer: {a}"
        example_strings = [
            TEMPLATE.format(q=e["question"], a=e["answer"]) for e in self.data
        ]
        return "\n".join(example_strings)

    def get_batch(self, split: Literal["train", "dev", "test"], n: int) -> list[dict]:
        """
        Randomly select n examples from a split.
        """
        assert split in ["train", "dev", "test"]
        data = self.__getattribute__(split)
        if n < len(data):
            return random.sample(data, n)
        return data
    
    def eval_on_batch(self, prompt: Prompt, batch: list[Example], phase="optim"):
        """
        Generates LLM solutions for a prompt on a Example batch and 
        uses the appropriate grading function where applicable.
        """
        grading_function = (
            self.grading_function if phase == "optim" else self.eval_function
        )

        # Do not regrade loaded prompts when continuing run
        if (
            grading_function is not None
            and prompt.get_score("dev") > -1.0
            and phase == "optim"
        ):
            logger.warning(
                f"Tried grading prompt {prompt.text} which is already graded."
            )
            return prompt.get_score("dev")

        # no batch_score for comparison based grading
        batch_score = None if grading_function is None else 0.0

        # workaround for imageurl 
        split = "dev" if self.answer_type=="imageurl" else batch[0].split 

        for i, example in enumerate(batch):
            llm_input = prompt.format(example.question)
            logger.debug(
                f"Grading problem id {example.id} on split {split} using prompt {prompt.id}"
            )
            response = self.solve_handle(question=llm_input)
            batch_score, grade = self.update_batch_score(response, grading_function, example.gold, batch_score)
            
            logger.debug(f"Problem {i + 1} score: {grade}")
            attempt = Attempt(prompt.id, example.id, response, grade)
            example.attempts.append(attempt)
            prompt.attempts.append(attempt)
            self.dump()

        # report average score
        if batch_score is not None:
            prompt.set_score(split, batch_score / len(batch))
        return prompt.get_score(split)

    def update_batch_score(self, response: tuple | str, grading_function: Callable, gold: str | int | tuple, batch_score: float) -> float:
        """
        Handles various responses, gets grade and updates batch score.
        """
        grade = None

        # if llm call/parsing was successful
        if response:
            if self.answer_type == "imageurl": # dalle has no reasoning
                solution = response[:]
                reasoning = None
                response = (None, solution)
            else:
                # check if solution dict was parsed
                reasoning, solution_dict = response
                solution = None if solution_dict is None else solution_dict["solution"]
            logger.debug(
                f"Got response with \nreasoning: {reasoning}\nsolution: {solution}"
            )

            # dont update for comparison-based eval
            if batch_score is not None:
                grade = grading_function(solution, gold)
                batch_score += grade

        return batch_score, grade