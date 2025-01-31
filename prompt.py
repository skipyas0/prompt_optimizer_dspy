import dspy
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/scores.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class Prompt:
    def __init__(self, text: str, gen: int):
        self.text = text
        self.gen = gen
        self.__dev_score = -1.0 if self.__valid() else 0.0
        self.__test_score = -1.0 if self.__valid() else 0.0

    def __valid(self) -> bool:
        """
        Valid prompts do not have additional formatting brackets.
        """
        valid = len(re.findall("{.*?}", str(self.text))) == 1 and '{}' in self.text    
        if not valid:
            logger.warning(f"Prompt '{self.text}' is invalid")
        return valid
    
    def __str__(self) -> str:
        return self.text
    
    def format(self, s: str) -> str:
        return self.text.format(s)

    def score(self, batch: list[dspy.Example], solve: dspy.Module, final: bool = False) -> float:
        if self.__dev_score >= 0.0 and not final:
            return self.__dev_score
        
        split_name = "TEST" if final else "DEV"
        batch_score = 0.0
        for i, example in enumerate(batch):
            question = self.format(example.question)
            solutions = solve(question=question).completions
            example = float(example.answer)
            avg_score_on_sample = 0.0
            N_SOLUTIONS = len(solutions.answer)
            logger.debug(f"Grading problem {i+1} on split {split_name}")
            for comp_idx in range(N_SOLUTIONS):
                reasoning = solutions.reasoning[comp_idx]
                solution = solutions.answer[comp_idx]
                try:
                    solution = float(solution)
                    grade = 1.0 if solution==example else 0.0
                except ValueError:
                    logger.warning(f"Couldn't convert '{solution}' to float")
                    grade = 0.0
                avg_score_on_sample += grade
                logger.debug(f"Completion {comp_idx+1}\nReasoning:{reasoning}\nSolution:{solution}\t|\tGold:{example}\nPass:{grade}\n")
            batch_score += avg_score_on_sample / N_SOLUTIONS

        if final:
            self.__test_score = batch_score / len(batch)
            return self.__test_score
        else:
            self.__dev_score = batch_score / len(batch)
            return self.__dev_score
        
    def jsoned(self) -> dict:
        return {"gen": self.gen, "prompt": str(self), "dev_score": self.__dev_score, "test_score": self.__test_score}