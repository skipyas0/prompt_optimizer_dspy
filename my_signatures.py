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
        return Signature(self.input_fields.copy(), self.output_fields.copy())

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
                f"{f.name}": {"type": str(f.type), "description": f.desc, "value": None}
                for f in self.input_fields
            },
            prefix
            + "outputs": {
                f"{f.name}": {
                    "type": str(f.type),
                    "description": f.desc,
                }
                for f in self.output_fields
            },
        }

    def matches_output(self, output: dict) -> bool:
        """
        Checks if output matches specification and in-place parses the values in the output dict if possible.
        """
        for field in self.output_fields:
            if field.name not in output.keys():
                return False
            if not field.type is type(output[field.name]):
                output[field.name] = utils.try_parse(output[field.name], field.type)
            if output[field.name] is None:
                return False
        outputs = self.mandatory_outputs()
        for key in output.keys():
            if key not in outputs:
                return False
        return True
    
lamarckian = Signature(
    [Field("examples", str, "")],
    [Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Given several input/output examples, design a suitable zero-shot prompt for a Large Language Model for that task.
    Before you create your prompt, **reflect** on these questions:
    - What is the nature of the task?
    - How general/specific should your prompt be?
    - Do the input/output examples belong to the same category or do you see any variations?
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

reflective = Signature(
    [
        Field("original_prompt", str, ""),
        Field("task_question", str, ""),
        Field("reasoning", str, "")
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

iterative = Signature(
    [Field("old_prompts", list, "")],
    [Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Given several prompts and their respective scores, propose new prompt.
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

encode = Signature(
    [Field("input_prompt", str, "")],
    [Field("characteristics", dict, "")],
    textwrap.dedent("""\
    Your supervisor tasked you with **describing** a piece of text, specifically a prompt for a Large Language Model.
    Look at the input prompt and write and describe it **briefly** (with a few words or numerically) for each of the following categories:
    - length (how long is the prompt, shorter/longer than necessary ...)
    - starting phrase
    - ending phrase
    - formatting brackets location ({})
    - writing_style (formal/informal, dry/interesting, ...)
    - author (imagine the person who could have written the prompt and describe them briefly)
    - theme (does the prompt try to put the reader into a specific setting, like some magical land etc.?)
    - identity (does the prompt identity assignment to the reader, like saying they are a proficient scientiest etc.?)
    - thinking_style (what style of thinking do the instructions promote in the reader?)
    - specificity (does the prompt go into specific step-by-step instructions or is it general?)
    - desired_outcome (what does the prompt ask for and in what way?)
    - examples_given  (does the prompt give any examples? what is their nature?)
    - reasoning (does the prompt induce Chain-of-Thought using a cue similar to "Let's think by step"?)
    - persona assignment (does the prompt utilize the persona assignment technique ("Imagine you are a skilled writer" etc.)?)
    - repetition (does the prompt have repetitive phrasing? is it excessive or could it be useful for emphasis?)
    Respond with your characteristics for the categories (length, structure, writing style, persona, thinking style, specificity, desired outcome, examples given)
    """)
)

lamarckian_decoder = Signature(
    [
        Field("examples", str, ""),
        Field("characteristics", dict, "")
    ],
    [Field("prompt_proposal", str, "")],
    textwrap.dedent("""\
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Given several input/output examples, design a suitable zero-shot prompt for a Large Language Model for that task according to the provided characteristics.
    Here is the overview of the characteristics provided to specify the nature of the prompt:
    - length (how long is the prompt, shorter/longer than necessary ...)
    - starting phrase
    - ending phrase
    - formatting brackets location ({})
    - writing_style (formal/informal, dry/interesting, ...)
    - author (imagine the person who could have written the prompt and describe them briefly)
    - theme (does the prompt try to put the reader into a specific setting, like some magical land etc.?)
    - identity (does the prompt identity assignment to the reader, like saying they are a proficient scientiest etc.?)
    - thinking_style (what style of thinking do the instructions promote in the reader?)
    - specificity (does the prompt go into specific step-by-step instructions or is it general?)
    - desired_outcome (what does the prompt ask for and in what way?)
    - examples_given  (does the prompt give any examples? what is their nature?)
    - reasoning (does the prompt induce Chain-of-Thought using a cue similar to "Let's think by step"?)
    - persona assignment (does the prompt utilize the persona assignment technique ("Imagine you are a skilled writer" etc.)?)
    - repetition (does the prompt have repetitive phrasing? is it excessive or could it be useful for emphasis?)
    Having reflected on these characteristics, **design your prompt**. 
    Keep in mind to only **exactly** one pair of brackets '{}' in your prompts to indicate where the question should be inserted.
    """)
)

analyst = Signature(
    [
        Field("question", str, ""),
        Field("context", dict, "")
    ],
    [Field("answer", str, "")],
    textwrap.dedent("""\
    You are an analyst in a prompt optimization process.
    Your supervisor asked you a question and you will answer it concisely based on the provided context.
    """)
)

optimization_success = Signature(
    [Field("introduction", str, "Optimization Director metaprompt.")],
    [Field("result", bool, "Was the optimization succesful?")],
    "Guide a prompt optimization procedure to achieve the best possible result."
)