import dspy

class InstructionInductor(dspy.Signature):
    """
    Given several input/output examples, design a suitable zero-shot prompt for a Large Language Model for that task.
    """
    examples: str = dspy.InputField()
    prefix_prompt: str = dspy.OutputField()
    suffix_prompt: str = dspy.OutputField()

class OptimizerIterator(dspy.Signature):
    """
    Given several prompts and their respective scores, propose new prompt.
    """
    old_prompts: list[tuple[str,float]] = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

