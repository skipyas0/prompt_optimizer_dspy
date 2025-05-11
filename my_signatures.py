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


reflective = Signature(
    [
        Field("original_prompt", str, "Improve this prompt"),
        Field("task_question", str, "Task on which the prompt was used"),
        Field("solution", str, "What the original prompt produced"),
    ],
    [Field("original_prompt_critique", str, "Faults in the original prompt"), Field("prompt_proposal", str, "Improved prompt")],
    textwrap.dedent("""\
    Improve a prompt for an LLM.
    
    You are an intelligent reflection function capable of advanced reasoning and prompt synthesis.
    Follow these steps to craft a better prompt:
    - Analyze the original prompt and its suboptimal performance on a task sample.
    - Find failure points in the solution and cross-reference to identify weaknesses in the prompt.
    - Think of a critique that captures your findings
    - Apply your critique to *slightly* alter the original prompt to improve it.
    Your improved prompt should still be **widely applicable and generic**.

    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    As a placeholder for the task question, '<INSERT TASK QUESTION HERE>' should be used exactly ONCE.
    In the final answer, do not include a title or any additional data, just the prompt.
    """)
)

iterative = Signature(
    [Field("old_prompts", list, "List of previous prompts with scores")],
    [Field("prompt_proposal", str, "Better prompt")],
    textwrap.dedent("""\
    Craft a new prompt for an LLM
    
    You are an intelligent pattern continuation function capable of advanced reasoning and prompt synthesis.
    You are given a given a history of past prompts along with their scores.
    They are listed in ascending order of fitness.
    Follow the sequence and design an improved prompt. 
    
    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    As a placeholder for the task question, '<INSERT TASK QUESTION HERE>' should be used exactly ONCE.
    In the final answer, do not include a title or any additional data, just the prompt.
    """)
)

lamarckian = Signature(
    [Field("task_examples", list, "Samples from a problem class")],
    [Field("prompt_proposal", str, "Instructions for solving the problem")],
    textwrap.dedent("""\
    Craft **general** developer prompt to help an LLM with solving a class of problems.
    
    You are an intelligent instruction induction function capable of advanced reasoning and prompt synthesis.
    Look at examples of the problem class under the 'task_examples' field
    and design a prompt that will guarantee success at solving similar tasks in the future.
    Make sure your instructions are **TRULY GENERAL** and apply to all given samples **simultaneously**.

    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    As a placeholder for the task question, '<INSERT TASK QUESTION HERE>' should be used exactly ONCE.
    In the final answer, do not include a title or any additional data, just the prompt.
    """)
)

lamarckian_personas = Signature(
    [Field("task_examples", list, "Samples from a problem class"), Field("persona", list, "Assume this persona when writing the prompt")],
    [Field("prompt_proposal", str, "Instructions for solving the problem")],
    textwrap.dedent("""\
    Craft **general** developer prompt to help an LLM with solving a class of problems.
    
    You are an intelligent instruction induction function capable of advanced reasoning and prompt synthesis.
    While crafting the prompt, you will assume the *persona* specified in the 'persona' field.
    Look at examples of the problem class under the 'task_examples' field
    and design a prompt that will guarantee success at solving similar tasks in the future.
    Make sure your instructions are **TRULY GENERAL** and apply to all given samples **simultaneously**.

    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    As a placeholder for the task question, '<INSERT TASK QUESTION HERE>' should be used exactly ONCE.
    In the final answer, do not include a title or any additional data, just the prompt.
    """)
)

lamarckian_values = Signature(
    [Field("task_examples", list, "Samples from a problem class"), Field("focus", list, "Values to focus on while writing the prompt")],
    [Field("prompt_proposal", str, "Instructions for solving the problem")],
    textwrap.dedent("""\
    Craft **general** developer prompt to help an LLM with solving a class of problems.
    
    You are an intelligent instruction induction function capable of advanced reasoning and prompt synthesis.
    You proud yourself in focusing on the *value* specified in the 'focus' field.
    Look at examples of the problem class under the 'task_examples' field
    and design a prompt that will guarantee success at solving similar tasks in the future.
    Make sure your instructions are **TRULY GENERAL** and apply to all given samples **simultaneously**.
                    
    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    As a placeholder for the task question, '<INSERT TASK QUESTION HERE>' should be used exactly ONCE.
    In the final answer, do not include a title or any additional data, just the prompt.
    """)
)

compare = Signature(
    [Field("task_question", str, ""),Field("prompt_a", str, ""), Field("output_a", str, ""),Field("prompt_b", str, ""), Field("output_b", str, "")],
    [Field("output_comparison", str, ""), Field("prompt_comparison", str, ""), Field("verdict", str, "")],
    textwrap.dedent("""\
    Compare performance of two prompts on a task.
                    
    You are an intelligent examiner capable of comparing two prompts and their outputs.
    You are given a task question and two prompts, "prompt_a" and "prompt_b", with their outputs.
    Follow these steps:
        1 - Understand the task question and think about how you would solve it.
        2 - Look at the outputs of both prompts and compare them.
        3 - Look at the prompts, understand their structure and how they relate to the outputs.
        4 - Write a detailed critique of the two prompts while reflecting on how they influence the outputs.
        5 - Make a final verdict on which prompt is better. Fill the "verdict" field with either "prompt_a" or "prompt_b".
""")
)

feedback = Signature(
    [Field("base_prompt", str, "Improve this prompt"),
    Field("comparisons", list, "Base prompt compared to others")],
    [Field("prompt_proposal", str, "Improved prompt")],
    textwrap.dedent("""\
        Improve a prompt for an LLM.

        You are an intelligent critique synthesis function capable of advanced reasoning. 
        You are given a base prompt and a list of comparisons between the base prompt and other prompts.
        Some other prompts are better than the base prompt, some are worse.
        Your task is to analyze the comparisons and synthesize a new prompt that incorporates the feedback.
        
        Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
        As a placeholder for the task question, '<INSERT TASK QUESTION HERE>' should be used exactly ONCE.
        In the final answer, do not include a title or any additional data, just the prompt.
        """)
)

paraphrase = Signature(
    [Field("original_prompt", str, "Prompt to paraphrase")],
    [Field("prompt_proposal", str, "Paraphrased prompt")],
    textwrap.dedent("""\
    Paraphrase a prompt for an LLM.
                    
    You are an intelligent paraphrasing function capable of advanced reasoning and prompt synthesis.
    You are given a prompt and your task is to paraphrase it. 
    Use synonyms and change the structure of the prompt but keep it semantically equivalent.

    Use markdown formatting in you final answer to indicate bullet points and whatever else necessary.
    As a placeholder for the task question, '<INSERT TASK QUESTION HERE>' should be used exactly ONCE.
    In the final answer, do not include a title or any additional data, just the prompt.
    """)
)

image_lamarckian = Signature(
    [Field("description", list, "Adjectives describing the image")],
    [Field("prompt_proposal", str, "Prompt for generating the image")],
    textwrap.dedent("""\
    Craft a prompt for a text-to-image model.
    You are a text-to-image model prompting expert, savvy in expressing emotions through visual arts.
    You are given a list of adjectives describing the image and your task is to create a prompt that will generate an image that matches the description.
    
    Make sure the prompt is clear and does not contain ambiguities.
    Keep your answer concise and make sure it fits the description.
    """)
)

image_feedback = Signature(
    [Field("base_prompt", str, "Improve this prompt"),
    Field("comparisons", list, "Base prompt compared to others")],
    [Field("prompt_proposal", str, "Improved prompt")],
    textwrap.dedent("""\
    Improve a prompt for an image generation model.

    You are an intelligent critique synthesis function capable of advanced reasoning. 
    You are given a base prompt and a list of comparisons between the base prompt and other prompts.
    Some other prompts are better than the base prompt, some are worse.
    Your task is to analyze the comparisons and synthesize a new prompt that incorporates the feedback.
    
    Make sure the prompt is clear and does not contain ambiguities.
    Keep your answer concise and make sure it fits the description.
    """)
)

image_compare = Signature(
    [Field("description", list, ""),Field("prompt_a", str, ""), Field("output_a", utils.imageurl, ""),Field("prompt_b", str, ""), Field("output_b", utils.imageurl, "")],
    [Field("output_comparison", str, ""), Field("prompt_comparison", str, ""), Field("verdict", str, "prompt_a or prompt_b")],
    textwrap.dedent("""\
    Compare images generated by two prompts.
                    
    You are an intelligent examiner capable of comparing two prompts and their image outputs.
    You are given a description of the image and two prompts, "prompt_a" and "prompt_b", with their outputs.
    First image is output_a and second image is output_b.
    
    Follow these steps:
        1 - Read the description.
        2 - Look at the output images and systematically determine which one better fits the description.
        3 - Look at the prompts, understand their structure and how they relate to the outputs.
        4 - Write a detailed critique of the two prompts while reflecting on how they influence the outputs.
        5 - Make a final verdict on which prompt is better. Fill the "verdict" field with either "prompt_a" or "prompt_b".
""")
)