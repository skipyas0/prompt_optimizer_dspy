import logging
import os
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
        id: str = None,
        placeholder: str = PLACEHOLDER,
    ):
        """
        Representation of a single prompt in the optimization population.
        """
        self.text = text
        self.gen = gen
        self.origin = origin
        self.__dev_score = -1.0 
        self.__test_score = -1.0
        self.active = active
        self.id = uuid.uuid4().hex if id is None else id
        self.attempts = []
        self.comparisons = []
        self.placeholder = placeholder
        self.sanitize()

    def sanitize(self) -> None:
        """
        Checks if the prompt has exactly 1 placeholder and fixes it if not.
        """
        if self.placeholder is not None:
            count = self.text.count(self.placeholder)
            if count == 0:
                logger.warning(f"Prompt {self.text} does not contain {self.placeholder}")
                self.text = self.text + self.placeholder
            elif count > 1:
                logger.warning(f"Prompt {self.text} contains multiple {self.placeholder}")
                self.text = self.text.replace(self.placeholder, "", count-1)
        

    def __str__(self) -> str:
        return self.text

    def format(self, s: str) -> str:
        return self.text if self.placeholder is None else self.text.replace(self.placeholder, s)

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

    def prompt_and_perf(self) -> tuple[str, float]:
        return (self.text, self.__dev_score)

    def get_score(self, split: Literal["dev", "test"]) -> float:
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
        """
        Adds comparison to prompt history and updates win rate.
        """
        comparison = comparison.copy()
        verdict = comparison["verdict"]
        winner_id = comparison[verdict] if verdict in ["prompt_a", "prompt_b"] else None

        # did I win?
        comparison["verdict"] = winner_id == self.id
        self.comparisons.append(comparison)

        # score update
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