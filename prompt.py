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
    def __init__(self, prefix: str, suffix: str, gen: int):
        self.prefix = prefix
        self.suffix = suffix
        self.text = prefix + suffix
        self.gen = gen
        valid = self.__valid()
        self.__dev_score = -1.0 if valid else 0.0
        self.__test_score = -1.0 if valid else 0.0

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

    def score(self, batch: list[dspy.Example], solve: dspy.Module, final: bool = False) -> float:
        if final:
            gs = lambda: self.__test_score
            def ss(x): self.__test_score = x
            split_name = "TEST"
        else: 
            gs = lambda: self.__dev_score
            def ss(x): self.__dev_score = x
            split_name = "DEV"

        if gs() >= 0.0:
            return gs()
            
        batch_score = 0.0
        for i, example in enumerate(batch):
            question = self.format(example.question)
            response = solve(question=question)
            if response:
                solutions = response.completions
                example = float(example.answer)
                avg_score_on_sample = 0.0
                N_SOLUTIONS = len(solutions.answer)
                logger.debug(f"Grading problem {i+1} on split {split_name}")
                for comp_idx in range(N_SOLUTIONS):
                    rationale = solutions.rationale[comp_idx]
                    solution = solutions.answer[comp_idx]
                    try:
                        solution = float(solution)
                        grade = 1.0 if solution==example else 0.0
                    except ValueError:
                        logger.warning(f"Couldn't convert '{solution}' to float")
                        grade = 0.0
                    avg_score_on_sample += grade
                    logger.debug(f"Completion {comp_idx+1}\nRationale:{rationale}\nSolution:{solution}\t|\tGold:{example}\nPass:{grade}\n")
                batch_score += avg_score_on_sample / N_SOLUTIONS

        ss(batch_score / len(batch))
        return gs()
        
    def jsoned(self) -> dict:
        return {"gen": self.gen, "prompt": str(self), "dev_score": self.__dev_score, "test_score": self.__test_score}
    
    def score_to_count(self) -> int:
        return round(self.__dev_score*10 + 3)
    
    def prompt_and_perf(self):
        return (self.text, self.__dev_score)