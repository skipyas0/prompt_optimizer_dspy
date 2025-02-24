import dspy

class Lamarckian(dspy.Signature):
    """
    Your supervisor tasked you with generating a prompt for a Large Language Model.
    Given several input/output examples, design a suitable zero-shot prompt for a Large Language Model for that task.
    Use curly brackets '{}' to indicate where the problem should be inserted.
    """
    examples: str = dspy.InputField()
    supervisor_hint: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

class Reflective(dspy.Signature):
    """
    Your supervisor tasked you with generating a prompt for a Large Language Model.
    Analyze a prompt and its suboptimal performance on a task sample along with its generated reasoning chain.
    Identify weak points and flaws in the prompt and think of a critique.
    The critique should answer the following questions:
    - Why does the original prompt get an incorrect answer?
    - What is the problem in the reasoning chain?
    - How does the prompt promote reasoning errors?
    - Does the prompt work for a general problem or is it too specific?
    Your task is to alter the original prompt to eliminate the problems from your critique.
    """
    original_prompt: str = dspy.InputField()
    task_question: str = dspy.InputField()
    reasoning: str = dspy.InputField()
    supervisor_hint: str = dspy.InputField()
    #prompt_critique: str = dspy.OutputField()
    prompt_proposal: str = dspy.OutputField()

class Iterative(dspy.Signature):
    """
    Your supervisor tasked you with generating a prompt for a Large Language Model.
    Given several prompts and their respective scores, propose new prompt.
    """
    old_prompts: list[tuple[str,float]] = dspy.InputField()
    supervisor_hint: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

class Crossover(dspy.Signature):
    """
    Your supervisor tasked you with generating a prompt for a Large Language Model.
    In the prompts field, you are given two distinct original prompts with their scores. 
    Your task is create a novel prompt taking inspiration from both original prompts.
    Try to combine the best elements from both original prompts to create the best offspring prompt.
    """
    prompt_a: tuple[str, float] = dspy.InputField()
    prompt_b: tuple[str, float] = dspy.InputField()
    supervisor_hint: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

class Analyst(dspy.Signature):
    """
    You are an analyst in a prompt optimization process.
    Your supervisor asked you a question and you will answer it concisely based on the provided context.
    """
    question: str = dspy.InputField()
    context: dict[str, str] = dspy.InputField()
    answer: str = dspy.OutputField()

class OptimizationSuccess(dspy.Signature):
    """Guide a prompt optimization procedure to achieve the best possible result."""
    introduction: str = dspy.InputField(desc="Optimization Director metaprompt.")
    result: bool = dspy.OutputField(desc="Was the optimization succesful?")