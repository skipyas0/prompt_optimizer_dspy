import signatures
import dspy
import logging
import os
from prompt import Prompt
from population import Population
import matplotlib.pyplot as plt
import random
import Levenshtein
from data import Data
import utils

CREATIVE_TEMP = 0.75

N_SOLUTIONS = 1

POP_SIZE = 100#8#20
ITER = 25

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/optim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class Optimizer:
    def __init__(self, data: Data):
        logger.info(f"Settings: {POP_SIZE=}, {ITER=}, {N_SOLUTIONS=}")
        self.data = data
        logger.info(f"Split lengths: train {len(self.data.train)}, dev {len(self.data.dev)}, test {len(self.data.test)}")
        self.start_gen = 1
        self.population = Population([])
        self.all_prompts = Population([])

        self.operators = {
            "LAMARCKIAN": self.lamarckian,
            "REFLECTIVE": self.reflective,
            "ITERATIVE": self.iterative,
            "CROSSOVER": self.crossover,
            "MUTATION": self.mutation,
            "LAMARCKIAN_DECODER": self.lamarckian_decoder,
            "RANDOM": self.random_op
        }

        op_type = os.environ["OPTIM_OP"]
        self.op = self.operators[op_type]
    def random_op(self, hint: str = "", gen: int = 0, n: int = 1):
        return random.choice(list(self.operators.values())[:4])(hint=hint, gen=gen, n=n)
    
    ### TOOLS FOR PROMPT GENERATION
    def lamarckian(self, hint: str = "", gen: int = 0, n: int = 1) -> None:
        """
        Task another component with access to the data to generate a new prompt.
        You may provide a short hint.

        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """
        N_EXAMPLES = 5
        module = dspy.ChainOfThought(signature=signatures.Lamarckian, temperature=CREATIVE_TEMP)
        for _ in range(n):
            prompt = module(examples=str(self.data.select('train', N_EXAMPLES))).prompt_proposal
            prompt_obj = Prompt(prompt, origin="lamarckian", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            self.population.update_tool_effectivity("lamarckian", score)
            logger.info(f"LAMARCKIAN generated prompt:\n {str(prompt_obj)}\nSCORE: {score}\nTools:{self.population.tool_effectivity}")

    def lamarckian_decoder(self, hint: str = "", gen: int = 0, n: int = 1) -> None:
        N_EXAMPLES = 3
        module = dspy.ChainOfThought(signature=signatures.LamarckianDecoder, temperature=CREATIVE_TEMP)
        for _ in range(n):
            char = random.choice(self.population.prompts).characteristics
            cats = random.sample(["length", "structure", "thinking_style", "writing_style", "persona", "theme", "identity"], 4)
            for cat in cats:
                desc = char[cat][:] # copy str
                if random.random() > 0.5:
                    mut = "antonymum"
                    char[cat] = utils.antonymum(phrase=desc, phrase_context=cat).opposite_meaning_phrase_in_context
                else:
                    mut = "twist"
                    char[cat] = utils.twist(phrase=desc, phrase_context=cat).phrase_with_unexpected_twist

                logger.info(f"Mutation {mut} changed description of category {cat} in decoder: {desc} -> {char[cat]}")
            prompt = module(examples=str(self.data.select('train', N_EXAMPLES)), characteristics=char).prompt_proposal
            prompt_obj = Prompt(prompt, origin="lamarckian_decoder", gen=gen, characteristics=char)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            self.population.update_tool_effectivity("lamarckian", score)
            logger.info(f"LAMARCKIAN_DECODER generated prompt:\n {str(prompt_obj)}\nSCORE: {score}\nTools:{self.population.tool_effectivity}")

    def reflective(self, hint: str = "", gen: int = 0, n: int = 1) -> None:
        """
        Task another component to improve an underperfoming prompt.
        The prompt is automatically chosen from the worst quartile.
        Your subordinate will write a critique of the prompt and try to write a better one.
        You may provide a short hint.


        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """

        module = dspy.ChainOfThought(signature=signatures.Reflective, temperature=CREATIVE_TEMP)
        for _ in range(n):
            completion = None
            while not completion:
                original = random.choice(self.population.prompts)
                completion = original.get_completion(0)
                logger.warning(f"Got completion {completion} in reflective, prompt {str(original)}")
            task = completion[0]
            reasoning = completion[1]
            completion = module(original_prompt=original.text, task_question=task.question, reasoning=reasoning)
            prompt = completion.prompt_proposal
            prompt_obj = Prompt(prompt, origin="reflective", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            logger.info(f"REFLECTIVE generated prompt:\n {str(prompt_obj)}\nSCORE: {score}.\nOriginal prompt:\n{original.text}\nTools:{self.population.tool_effectivity}")
            self.population.update_tool_effectivity("reflective", score)
    
    def iterative(self, hint: str = "", gen: int = 0, n: int = 1) -> None:
        """
        Task another component to create a new prompt based on several prompt+score examples.
        The examples are chosen automatically using roulette selection with the best prompts having the biggest chance of being featured.
        You may provide a short hint.


        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """

        module = dspy.ChainOfThought(signature=signatures.Iterative, temperature=CREATIVE_TEMP)
        for _ in range(n):
            examples = [p.prompt_and_perf() for p in self.population.select(5)]
            prompt = module(old_prompts=examples).prompt_proposal
            prompt_obj = Prompt(prompt, origin="iterative", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            logger.info(f"ITERATIVE generated prompt:\n {str(prompt_obj)}\nSCORE: {score}.\nExamples:\n{examples}\nTools:{self.population.tool_effectivity}")
            self.population.update_tool_effectivity("iterative", score)

    def crossover(self, hint: str = "", gen: int = 0, n: int = 1) -> Prompt:
        """
        Task another component to create a new prompt using two diverse parent prompts.
        The prompts are chosen automatically.
        You may provide a short hint.


        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """
        
        module = dspy.ChainOfThought(signature=signatures.Crossover, temperature=CREATIVE_TEMP)
        for _ in range(n):
            best_quartile = self.population.quartile(1)
            prompt1 = random.choice(best_quartile) if len(best_quartile) > 1 else self.population.prompts[0]
            # prompt2 most distinct to prompt1
            prompt2 = sorted(self.population, key= lambda p: Levenshtein.distance(prompt1.text, p.text))[-1]
            prompts = [prompt1, prompt2]
            random.shuffle(prompts)
            prompt = module(prompt_a=prompts[0].prompt_and_perf(), prompt_b=prompts[1].prompt_and_perf()).prompt_proposal
            prompt_obj = Prompt(prompt, origin="crossover", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            logger.info(f"CROSSOVER generated prompt:\n {str(prompt_obj)}\n SCORE: {score}\nfrom prompt1:\n{prompts[0]}\n and from prompt2:\n{prompts[1]}\nTools:{self.population.tool_effectivity}.")
            self.population.update_tool_effectivity("crossover", score)

    def mutation(self, hint: str = "", gen: int = 0, n: int = 1):
        """
        Task another component to paraphrase a prompt.
        The prompts are chosen automatically.
        You may provide a short hint.

        Args:
            hint (str): Specific instruction on how a new prompt should be constructed.
            gen (int): Current generation.
            n (int): How many times to repeat the operation

        Returns:
            None
        """
        module = dspy.ChainOfThought(signature=signatures.Mutation, temperature=CREATIVE_TEMP)
        for _ in range(n):
            input_prompt = random.choice(self.population.prompts)
            prompt = module(input_prompt=input_prompt).prompt_proposal
            prompt_obj = Prompt(prompt, origin="mutation", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            logger.info(f"MUTATION generated prompt:\n {str(prompt_obj)}\n SCORE: {score}\nfrom prompt:{input_prompt}\nTools:{self.population.tool_effectivity}.")
            self.population.update_tool_effectivity("mutation", score)


    def __run(self):
        for step in range(self.start_gen+1,self.start_gen+ITER+1):
            self.all_prompts.set_update(self.population.prompts)
            self.all_prompts.dump()
            n = self.population.purge_duplicates()
            self.op(gen=step, n = n)
            self.population.dump(gen=step)
        # add last generation
        self.all_prompts.set_update(self.population.prompts)

    def begin(self, initial_population: list[Prompt]=[]): 
        if len(initial_population) > 0:
            self.all_prompts.set_update(initial_population)
            active = [p for p in self.all_prompts if p.active]
            # score prompts with uninitialized dev scores 
            _ = [self.data.eval_on_split(p) for p in active if p.get_score("dev") == -1.0]
            self.population.add(active)
            self.start_gen = max([p.gen for p in self.population])
        if len(initial_population) < POP_SIZE:
            DECODER_POP_TH = 10
            remaining_simple_lamarck = DECODER_POP_TH - len(initial_population) 
            remaining_total = POP_SIZE - len(initial_population) 
            if remaining_simple_lamarck > 0: 
                self.lamarckian(n = remaining_simple_lamarck)
                remaining_total -= remaining_simple_lamarck 
            self.lamarckian_decoder(n = remaining_total)
            self.all_prompts.set_update(self.population.prompts)
        self.population.dump(gen=0)
        self.all_prompts.dump()

        logger.info("Starting optimization")
        self.__run()
        logger.info("Optimization done")
        self.all_prompts.dump()

    def eval(self):
        logger.info("Starting final eval")
        by_gen = self.all_prompts.evaluate_iterations(self.data)
        logger.info("Final eval done")
        self.all_prompts.dump()

        x = list(range(len(by_gen)))
        y_avg = [sum(g)/len(g) for g in by_gen]
        y_max = [max(g) for g in by_gen]
        logger.info(f"Evaluation stats:\nAvg: {' '.join(map(str,y_avg))},\nMax: {' '.join(map(str,y_max))}")
        plt.figure()
        plt.title("OPRO-like Hill-Climber")
        plt.plot(x, y_avg, color="blue", label="Average")
        plt.plot(x, y_max, color="red", label="Max")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Average score")
        plt.savefig(f"{os.getenv('RUN_FOLDER')}/plt.svg")



