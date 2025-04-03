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
from model_api import model
import my_signatures as sig
CREATIVE_TEMP = 0.75

N_SOLUTIONS = 1

POP_SIZE = 100#8#20
ITER = 25

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("evooptim.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class EvoOptimizer:
    def __init__(self, data: Data):
        logger.info(f"Settings: {POP_SIZE=}, {ITER=}, {N_SOLUTIONS=}")
        self.data = data
        logger.info(f"Split lengths: train {len(self.data.train)}, dev {len(self.data.dev)}, test {len(self.data.test)}")
        self.start_gen = 1
        self.population = Population([])
        self.all_prompts = Population([])

        self.operators = {
            "LAMARCKIAN": self.lamarckian,
            "CROSSOVER": self.crossover,
            "LAMARCKIAN_DECODER": self.lamarckian_decoder,
        }

        op_type = os.environ["OPTIM_OP"]
        self.op = self.operators[op_type]
    
    ### TOOLS FOR PROMPT GENERATION
    def lamarckian(self, gen: int = 0, n: int = 1) -> None:
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
        for _ in range(n):
            prompt = model.chain_of_thought(sig.lamarckian,examples=self.data.select('train', N_EXAMPLES))["prompt_proposal"]
            prompt_obj = Prompt(prompt, origin="lamarckian", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            self.population.update_tool_effectivity("lamarckian", score)
            logger.info(f"LAMARCKIAN generated prompt:\n {str(prompt_obj)}\nSCORE: {score}\nTools:{self.population.tool_effectivity}")

    def lamarckian_decoder(self, gen: int = 0, n: int = 1) -> None:
        N_EXAMPLES = 3
        for _ in range(n):
            char = random.choice(self.population.prompts).characteristics
           
            prompt = model.chain_of_thought(sig.lamarckian_decoder,examples=self.data.select('train', N_EXAMPLES), characteristics=char)["prompt_proposal"]
            prompt_obj = Prompt(prompt, origin="lamarckian_decoder", gen=gen, characteristics=char)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            self.population.update_tool_effectivity("lamarckian_decoder", score)
            logger.info(f"LAMARCKIAN_DECODER generated prompt:\n {str(prompt_obj)}\nSCORE: {score}\nTools:{self.population.tool_effectivity}")


    def crossover(self, gen: int = 0, n: int = 1) -> Prompt:
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
        #twist = sig.Signature.from_str(
        #    "phrase: str ();; phrase_context: str ()-> changed_phrase: str (rephrased phrase with an unexpected twist)"
        #)
        mutation = sig.Signature.from_str(
            "phrase: str ();; phrase_context: str ()-> changed_phrase: str (context-appropriate **mutated** version of the original phrase)"
        )
        mutation_rate = 0.2
        for _ in range(n):
            best_quartile = self.population.quartile(1)
            prompt1 = random.choice(best_quartile) if len(best_quartile) > 1 else self.population.prompts[0]
            # prompt2 most distinct to prompt1
            prompt2 = sorted(self.population, key= lambda p: Levenshtein.distance(prompt1.text, p.text))[-1]
            prompts = [prompt1, prompt2]
            random.shuffle(prompts)

            # offspring prompt characteristics
            offspring_char = dict()
            for aspect in prompt1.characteristics.keys():
                context = [f.desc for f in sig.lamarckian_decoder.output_fields if f == aspect][0]
                aspect_value = prompt1[aspect] if random.random() > 0.5 else prompt2[aspect]
                after_mutation = aspect_value if random.random() > mutation_rate else model.predict(
                        mutation, phrase=aspect_value, phrase_context=f"{aspect}: {context}")["changed_phrase"]
                offspring_char[aspect] = after_mutation

            prompt = model.chain_of_thought(sig.crossover_char, prompt_a=prompts[0].text, prompt_a_characteristics=prompts[0].characteristics, prompt_b=prompts[1].text, prompt_b_characteristics=prompts[1].characteristics, prompt_proposal_characteristics=offspring_char)["prompt_proposal"]
            prompt_obj = Prompt(prompt, origin="crossover", gen=gen)
            score = self.data.eval_on_split(prompt_obj) 
            self.population.add(prompt_obj)
            logger.info(f"CROSSOVER generated prompt:\n {str(prompt_obj)}\n SCORE: {score}\nfrom prompt1:\n{prompts[0]}\n and from prompt2:\n{prompts[1]}\nTools:{self.population.tool_effectivity}.")
            self.population.update_tool_effectivity("crossover", score)

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



