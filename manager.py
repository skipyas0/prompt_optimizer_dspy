import dspy
from prompt import Prompt
from population import Population
from data import Data
import random
import logging
import os
from typing import Literal, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


### ADIO Agent DIrected Optimization 

class OptimizationSuccess(dspy.Signature):
    """Guide a prompt optimization procedure to achieve the best possible result."""
    introduction: str = dspy.InputField(desc="Optimization Director metaprompt.")
    result: bool = dspy.OutputField(desc="Was the optimization succesful?")




class DirectedOptimizer:
    def __init__(self, teacher_lm: dspy.LM, solver_lm: dspy.LM, training_data: Data, population: Optional[Population] = None):
    
        # use teacher lm for all optimization related tasks
        self.teacher_lm = teacher_lm
        dspy.configure(lm=self.teacher_lm)

        # solver lm is the one for which we are optimizing
        self.solver_lm = solver_lm

        # solver lm wrapper
        solve_module = dspy.ChainOfThought("question: str -> answer: float", temperature=0.0, n=1)
        def solve(prompt: Prompt, example: dspy.Example):
            question = prompt.format(example.question)
            ret = None
            try:
                with dspy.context(lm=solver_lm):
                    ret = solve_module(question=question).completions
            except ValueError as e:
                logger.error(f"Model/COT error '{str(e)}' on solve, aborting solve")

            # add completion to the prompts' history for future llm analysis
            prompt.completions += (example, ret)
            return ret
            
        # handle passed to scoring functions
        self.solve = solve

        self.population = population if population else Population([], self.solve)

        self.training_data = training_data
        self.metaprompt = """You are the directing component in a prompt optimization procedure. 
        Use the provided tools to get information about the underlying task, inspect and modify the prompt population and check remaining budget.
        You are the manager of this project so delegate as much of the tasks to your subordinates using the provided tools."""
        self.generation = 0


        ### TOOLS
        def peek_data() -> str:
            """
            Show 3 random samples from the dataset in a formatted Q/A template.
            """
            return str(Data(random.sample(self.data, 3)))

        def peek_pop() -> str:
            """
            Show the total number of prompts in population and 3 random samples with their scores.
            """
            n = len(self.population.prompts)
            formatter = lambda prompt, perf: f"Prompt:\n```{prompt}\n```" + f"\nhas score {perf}" if perf >= 0 else "\nhasn't been scored yet"
            samples_text = '\n'.join(map(formatter, [p.prompt_and_perf() for p in random.sample(self.population, 3)]))
            return f"Population has {n} prompts\nSamples:\n{samples_text}"
        
        
        def lamarckian(hint: str = "") -> tuple[str, float]:
            """
            Task another component to look at input/output data examples and generate a new prompt.
            You may provide a short hint.
            This tool returns the resulting prompt and its score.
            """
            

        def reflective(target_section: Literal["best", "worst", "random"]) -> tuple[str, float]:
            """
            Task another component to select one prompt according to 'target_section' and try to improve it.
            Your subordinate will write a critique of the prompt and try to write a better one.
            This tool returns the resulting prompt and its score.
            """
            pass

        def iterative(hint: str = "") -> tuple[str, float]:
            """
            Task another component to create a new prompt based on several prompt+score examples.
            The examples are chosen using a roulette selection with the best prompts having the biggest chance of being featured.
            You may provide a short hint.
            This tool returns the resulting prompt and its score.
            """
            pass

        def crossover(ode: Literal["bert", "levenshtein"] = "levenshtein", hint: str = "") -> tuple[str, float]:
            """
            Task another component to create a new prompt using two parent prompts and their scores.
            First prompt has good performance.
            Second prompt is the one with the biggest distance from the first prompt.
            Two modes: cosine semantic similarity (bert), editing distance (levensthein)
            You may provide a short hint.
            This tool returns the resulting prompt and its score.
            """
            pass

        def purge_worst() -> None:
            """
            Remove the worst quarter (1/4) of prompts from the population.
            Begins new generation.
            """
            pass

        def purge_duplicates(mode: Literal["bert", "levenshtein"] = "levenshtein") -> None:
            """
            After sorting by score, go prompt by prompt and remove the most similar prompt until a quarter (1/4) of the population is deleted.
            Two modes: cosine semantic similarity (bert), editing distance (levensthein)
            Begins new generation.
            """
            pass

        def ask_analyst(question_index: Literal[0,1,2,3,4,5]) -> str:
            """
            Task your analyst to look at relevant data and use it to answer one of the following questions.
                0: "Can we optimize further or should I finish?"
                1: "Are the prompts diverse enough? How can I promote diversity?"
                2: "What's the most common problem in the prompts?"
                3: "What reasoning errors do some prompts promote?"
                4: "Is my population size optimal?"
                5: "Which operations create the best prompts?"
            """
    
            pass

    def run(self):
        return dspy.React(OptimizationSuccess, self.tools)(introduction=self.metaprompt)
    
