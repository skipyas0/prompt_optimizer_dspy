from __future__ import annotations
import re
import utils
import textwrap


class Field:
    def __init__(self, name: str, type: type, desc: str):
        self.name = name
        self.type = type
        self.desc = desc


class Signature:
    def __init__(
        self,
        input_fields: list[Field],
        output_fields: list[Field],
        instructions: str = "",
    ):
        self.instructions = instructions
        self.input_fields = input_fields
        self.output_fields = output_fields

    def copy(self) -> Signature:
        return Signature(self.input_fields.copy(), self.output_fields.copy(), self.instructions[:])

    def mandatory_inputs(self) -> list[str]:
        return [f.name for f in self.input_fields]

    def mandatory_outputs(self) -> list[str]:
        return [f.name for f in self.output_fields]

    def update_inputs(self, new, beg=True) -> None:
        inputs = self.mandatory_inputs()
        for field in new:
            if field.name in inputs:
                raise ValueError(f"Field {field.name} already in input fields")
            if beg:
                self.input_fields.insert(0, field)
            else:
                self.input_fields.append(field)

    def update_outputs(self, new, beg=True) -> None:
        outputs = self.mandatory_outputs()
        for field in new:
            if field.name in outputs:
                raise ValueError(f"Field {field.name} already in output fields")
            if beg:
                self.output_fields.insert(0, field)
            else:
                self.output_fields.append(field)

    @classmethod
    def from_str(cls, string_signature) -> Signature:
        """
        Parses a text signature of the format:
            inp1_name: inp1_type (inp1_desc);; ...;; inpn_name: inpn_type (inpn_desc)
            ->
            out1_name: out1_type (out1_desc);; ...;; outn_name: outn_type (outn_desc)
        """
        inputs_outputs = string_signature.split("->")
        assert (
            len(inputs_outputs) == 2
        ), f"Wrong signature format in '{string_signature}'"
        io_fields = [[], []]
        for source, fields in zip(inputs_outputs, io_fields):
            inputs = source.split(";;")
            for inp in inputs:
                match = re.match(r"(.+): (.+) \((.*)\)", inp.strip())
                if match and len(match.groups()) == 3:
                    name, type_str, desc = match.groups()
                    actual_type = utils.str_to_type(type_str)
                    fields.append(Field(name, actual_type, desc))
                else:
                    raise ValueError(f"Wrong signature format in '{string_signature}'")
        return Signature(io_fields[0], io_fields[1])

    def as_dict(self, prefix="") -> dict:
        # Optionally adds instructions field to inputs and outputs
        # Prefix aids differentiating between main signature and a context signature in multi-turn settings
        instructions = (
            {prefix + "instructions": self.instructions}
            if len(self.instructions) > 0
            else {}
        )
        # Dict join operator
        return instructions | {
            prefix
            + "inputs": {
                f"{f.name}": None
                for f in self.input_fields
            },
            prefix
            + "outputs": {
                f"{f.name}": {
                    "type": f.type.__name__,
                    "description": f.desc,
                }
                for f in self.output_fields
            },
        }

    def matches_output(self, output: dict) -> bool:
        """
        Checks if output matches specification and in-place parses the values in the output dict if possible.
        """
        # Sometimes the model wraps all outputs into an 'outputs' field
        if "outputs" in output.keys() and self.matches_output(output["outputs"]):
            output = output["outputs"]
            return True
        
        for field in self.output_fields:
            if field.name not in output.keys():
                return False
            #if not isinstance(output[field.name], field.type):
            #    print("before try parse:", field.name, field.type, '\n', output[field.name])
            #    output[field.name] = utils.try_parse(output[field.name], field.type)
            if output[field.name] is None:
                return False
        outputs = self.mandatory_outputs()
        for key in output.keys():
            if key not in outputs:
                return False
        return True
    
lamarckian = Signature(
    [Field("task_examples", list, "")],
    [Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Given several task examples, design a suitable zero-shot prompt for a Large Language Model for that task.
    Before you create your prompt, **reflect** on these questions:
    - What is the nature of the task?
    - How general/specific should your prompt be?
    - Do the task examples belong to the same category or do you see any variations?
    - What type if thinking is necessary for solving the problem?
    - Would your prompt benefit from including examples of the problem?
    - *IMPORTANT* Where will the question be inserted into your prompt? *HINT*: Only use a single pair of brackets '{}' in your prompt.
    - Can you make a step-by-step guide for solving the problem?
    - How would you solve the problem?
    - How can I give my own twist to the prompt so that it is **interesting** to the reader? 
    Having reflected on these questions, **design your prompt**. 
    Keep in mind to only **exactly** one pair of brackets '{}' in your prompts to indicate where the question should be inserted.
    """)
)

reflective1 = Signature(
    [
        Field("original_prompt", str, ""),
        Field("task_question", str, ""),
        Field("solution", str, "")
    ],
    [Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Analyze a prompt and its suboptimal performance on a task sample along with its generated reasoning chain.
    Identify weak points and flaws in the prompt and think of a critique.
    The critique should answer the following questions:
    - Why does the original prompt get an incorrect answer?
    - What is the problem in the reasoning chain?
    - How does the prompt promote reasoning errors?
    - Does the prompt work for a general problem or is it too specific?
    Your task is to alter the original prompt to eliminate the problems from your critique.
    """)
)
reflective2 = Signature(
    [
        Field("original_prompt", str, ""),
        Field("task_question", str, ""),
        Field("solution", str, "")
    ],
    [Field("original_prompt_critique", str, ""), Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Improve a prompt for an LLM.
    
    You are an intelligent reflection function capable of advanced reasoning and prompt synthesis.
    Follow these steps to craft a better prompt:
    - Analyze the original prompt and its suboptimal performance on a task sample.
    - Find failure points in the solution and cross-reference to identify weaknesses in the prompt.
    - Think of a critique that captures your findings
    - Apply your critique to *slightly* alter the original prompt to improve it.
    Your improved prompt should still be **widely applicable and generic**.

    Maintain the same formatting as in the original prompt.  
    In the final answer, do not include a title or any additional data, just the prompt.
    """)
)

iterative = Signature(
    [Field("old_prompts", list, "")],
    [Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Craft a new prompt for an LLM
    
    You are given a given a history of past prompts along with their scores.
    They are listed in ascending order of fitness.
    Follow the sequence and design an improved prompt. 
    """)
)

crossover = Signature(
    [
        Field("prompt_a", tuple, ""),
        Field("prompt_b", tuple, "")
    ],
    [Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    In the prompts field, you are given two distinct original prompts with their scores. 
    Your task is create a novel prompt taking inspiration from both original prompts.
    Try to combine the best elements from both original prompts to create the best offspring prompt.
    """)
)


lamarckian1 = Signature(
    [Field("task_examples", list, "Samples from problem class")],
    [Field("instruction_proposal", str, "Instructions for solving the problem")],
    textwrap.dedent("""\
    Create a general step-by-step instruction to help the user solve a class of problems.
    
    You are a wise advisor with general knowledge about many tasks.
    Look at examples of the problem class under the 'task_examples' field
    and design a tutorial that will guarantee the user's success at solving similar tasks in the future.
    Make sure your instructions are general and apply to all given samples simultaneously.
    
    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    """)
)

lamarckian2 = Signature(
    [Field("task_examples", list, "Samples from problem class")],
    [Field("instruction_proposal", str, "Instructions for solving the problem")],
    textwrap.dedent("""\
    Create a **general** step-by-step instruction to help the user solve a class of problems.
    
    You are a wise advisor with general knowledge about many tasks.
    Look at examples of the problem class under the 'task_examples' field
    and design a tutorial that will guarantee the user's success at solving similar tasks in the future.
    Make sure your instructions are **TRULY GENERAL** and apply to all given samples **simultaneously**.
                    
    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    """)
)


lamarckian3 = Signature(
    [Field("task_examples", list, "Samples from a problem category")],
    [Field("instruction_proposal", str, "Instructions for solving a different problem of the same category")],
    textwrap.dedent("""\
    Create a **general** step-by-step instruction to help the user solve a category of problems.
    
    You are a wise advisor with general knowledge about many tasks.
    Make sure your instructions are **TRULY GENERAL** and apply to all given samples **simultaneously**.
                    
    Follow these steps to make sure your answer is worthy:
        1 - Look at ALL examples in the 'task_examples' field.
        2 - Identify common elements, find the task category.
        3 - Create a step-by-step tutorial that applies to ALL the examples. 
        4 - Look over your tutorial to make sure it is truly general and helpful.
        5 - Write the final step-by-step instruction     
                    
    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    """)
)

lamarckian4 = Signature(
    [Field("task_examples", list, "Samples from a problem category"), Field("focus", list, "Values to focus on")],
    [Field("instruction_proposal", str, "Instructions for solving a different problem of the same category")],
    textwrap.dedent("""\
    Create a **general** step-by-step instruction to help the user solve a category of problems.
                    
    Follow these steps to make sure your answer is worthy:
        1 - Look at ALL examples in the 'task_examples' field.
        2 - Identify common elements, find the task category.
        3 - Create a step-by-step tutorial that applies to ALL the examples. 
        4 - Look over your tutorial to make sure it is truly general and helpful.
        5 - Write the final step-by-step instruction     
    """)
)

lamarckian5 = Signature(
    [Field("task_examples", list, "Samples from a problem class"), Field("focus", list, "Values to focus on")],
    [Field("instruction_proposal", str, "Instructions for solving the problem")],
    textwrap.dedent("""\
    Create a **general** step-by-step instruction to help the user solve a class of problems.
    
    You are a wise advisor with general knowledge about many tasks.
    Look at examples of the problem class under the 'task_examples' field
    and design a tutorial that will guarantee the user's success at solving similar tasks in the future.
    Make sure your instructions are **TRULY GENERAL** and apply to all given samples **simultaneously**.
                    
    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    """)
)

lamarckian6 = Signature(
    [Field("task_examples", list, "Samples from a problem class"), Field("focus", list, "Values to focus on")],
    [Field("prompt_proposal", str, "Instructions for solving the problem")],
    textwrap.dedent("""\
    Craft **general** developer prompt to help an LLM with solving a class of problems.
    
    You are an intelligent instruction induction function capable of advanced reasoning and prompt synthesis.
    Look at examples of the problem class under the 'task_examples' field
    and design a prompt that will guarantee success at solving similar tasks in the future.
    Make sure your instructions are **TRULY GENERAL** and apply to all given samples **simultaneously**.

    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    
    In the final answer, do not include a title or any additional data, just the prompt.
    """)
)

mutation = Signature(
    [Field("input_prompt", str, "")],
    [Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Specifically, your task is to take a look at the input prompt and **paraphrase** it.
    Here are some ways to do that:
    - Use fitting synonyms to conserve meaning and produce a semantically equivalent prompt.
    - Imagine you are writing a story and change the prompt to fit the narrative.
    - Add some of your reasoning to the prompt, particularly if the prompt includes examples where the answer is provided without explanation.
    Try to be original so that your prompt is fresh and interesting while still having all the instructional value.
    """)
)