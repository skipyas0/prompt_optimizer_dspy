import logging
import os
import re
from typing import Literal
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

PLACEHOLDER = '<INSERT TASK QUESTION HERE>'

class Prompt:
    def __init__(
        self,
        text: str,
        gen: int = 0,
        origin: str = "unknown",
        active: bool = True,
        id: str = None
    ):
        self.text = text
        self.gen = gen
        self.origin = origin
        self.valid = True
        self.sanitize()
        self.__dev_score = -1.0 if self.valid else 0.0
        self.__test_score = -1.0 if self.valid else 0.0
        self.active = active
        self.id = uuid.uuid4().hex if id is None else id
        self.attempts = []
        self.comparisons = []

    def sanitize(self) -> bool:
        """
        Sanitize prompt and return if it's valid
        Valid prompts do not have additional formatting brackets.
        """
        count = self.text.count(PLACEHOLDER)
        if count == 0:
            logger.warning(f"Prompt {self.text} does not contain {PLACEHOLDER}")
            self.text = self.text + PLACEHOLDER
        elif count > 1:
            logger.warning(f"Prompt {self.text} contains multiple {PLACEHOLDER}")
            self.text = self.text.replace(PLACEHOLDER, "", count-1)
        

    def __str__(self) -> str:
        return self.text

    def format(self, s: str) -> str:
        return self.text.replace(PLACEHOLDER, s)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "gen": self.gen,
            "prompt": self.text,
            "dev_score": self.__dev_score,
            "test_score": self.__test_score,
            "origin": self.origin,
            "active": self.active,
        }

    @classmethod
    def from_json(cls, prompt: dict):
        p = Prompt(
            prompt["prompt"], prompt["gen"], prompt["origin"], prompt["active"], prompt["id"]
        )
        p.__dev_score = prompt["dev_score"]
        p.__test_score = prompt["test_score"]
        return p

    def score_to_count(self) -> int:
        return round(self.__dev_score * 10 + 3)

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
        print(f"Setting score {score} for {self.text} on {split} from {old_score}")

        if split == "dev":
            self.__dev_score = score
        else:
            self.__test_score = score

    def update_comparisons(self, comparison):
        comparison = comparison.copy()
        winner = (
                comparison["prompt_a"]
                if comparison["verdict"] == "prompt_a"
                else comparison["prompt_b"]
            )
        comparison["verdict"] = winner == self.id
        self.comparisons.append(comparison)
        dev_comps = [
            c for c in self.comparisons if c["split"] == "dev" 
        ]
        if len(dev_comps) > 0:
            self.set_score("dev", sum([c["verdict"] for c in dev_comps]) / len(dev_comps))
        test_comps = [
            c for c in self.comparisons if c["split"] == "test"
        ]
        if len(test_comps) > 0:
            self.set_score("test", sum([c["verdict"] for c in test_comps]) / len(test_comps))