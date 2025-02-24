import dspy
import logging
import os
import re
from typing import Literal
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/scores.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class Prompt:
    def __init__(self, prefix: str, suffix: str = "", gen: int = 0, origin: str = "unknown", active: bool = True):
        self.prefix = prefix
        self.suffix = suffix
        self.text = prefix + suffix
        self.gen = gen
        self.origin = origin
        self.valid = self.__valid()
        self.__dev_score = -1.0 if self.valid else 0.0
        self.__test_score = -1.0 if self.valid else 0.0
        self.completions = []
        self.active = True

    def __valid(self) -> bool:
        """
        Sanitize prompt and return if it's valid
        Valid prompts do not have additional formatting brackets.
        """
        sanitized = re.sub("{.*?}", '{}', self.text)
        brackets_left = len(re.findall("{.*?}", str(sanitized)))
        if brackets_left == 1:
            self.text = sanitized
            valid = True
        elif brackets_left == 0:
            self.text = self.prefix+'{}'+self.suffix
            valid = True
        else:
            valid = False 
        self.text = self.text.replace('\"', '')
        valid = valid and len(re.findall("{[^}]|[^{]}", str(self.text))) == 0 
        if not valid:
            logger.warning(f"Prompt '{self.text}' is invalid")
        return valid
    
    def __str__(self) -> str:
        return self.text
    
    def format(self, s: str) -> str:
        return self.text.format(s)
    
    def get_completion(self, grade: Literal[0, 1]):
        if len(self.completions) == 0:
            return None
        try:
            # get a wrong completion
            completion = next(filter(lambda c: c[2] == grade, self.completions))
        except StopIteration:
            # get any completion
            completion = self.completions[-1]
        return completion
        
    def jsoned(self) -> dict:
        return {"gen": self.gen, "prompt": str(self), "dev_score": self.__dev_score, "test_score": self.__test_score, "origin": self.origin, "active": self.active}
    
    @classmethod
    def from_json(cls, prompt: dict):
        p = Prompt(prompt["prompt"], "", prompt["gen"], prompt["origin"], prompt["active"])
        p.__dev_score = prompt["dev_score"]
        p.__test_score = prompt["test_score"]
        return p

    def score_to_count(self) -> int:
        return round(self.__dev_score*10 + 3)
    
    def prompt_and_perf(self):
        return (self.text, self.__dev_score)
    
    def get_score(self, split: Literal["dev", "test"]):
        if split == "dev":
            return self.__dev_score
        elif split == "test":
            return self.__test_score
        else:
            raise ValueError(f"Wrong split {split}")
    
    def set_score(self, split: Literal["dev", "test"], score):
        old_score = self.get_score(split)
        if old_score == -1.0:
            if split == "dev":
                self.__dev_score = score
            else:
                self.__test_score = score
        else:
            raise ValueError(f"Prompt already has score {old_score}")