import dspy

class Lamarckian(dspy.Signature):
    """
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
    """
    examples: str = dspy.InputField()
    #supervisor_hint: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

class Reflective(dspy.Signature):
    """
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
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
    #supervisor_hint: str = dspy.InputField()
    #prompt_critique: str = dspy.OutputField()
    prompt_proposal: str = dspy.OutputField()

class Iterative(dspy.Signature):
    """
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Given several prompts and their respective scores, propose new prompt.
    """
    old_prompts: list[tuple[str,float]] = dspy.InputField()
    #supervisor_hint: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

class Crossover(dspy.Signature):
    """
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    In the prompts field, you are given two distinct original prompts with their scores. 
    Your task is create a novel prompt taking inspiration from both original prompts.
    Try to combine the best elements from both original prompts to create the best offspring prompt.
    """
    prompt_a: tuple[str, float] = dspy.InputField()
    prompt_b: tuple[str, float] = dspy.InputField()
    #supervisor_hint: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

class Mutation(dspy.Signature):
    """
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Specifically, your task is to take a look at the input prompt and **paraphrase** it.
    Here are some ways to do that:
    - Use fitting synonyms to conserve meaning and produce a semantically equivalent prompt.
    - Imagine you are writing a story and change the prompt to fit the narrative.
    - Add some of your reasoning to the prompt, particularly if the prompt includes examples where the answer is provided without explanation.
    Try to be original so that your prompt is fresh and interesting while still having all the instructional value.
    """
    input_prompt: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()
    
class Encode(dspy.Signature):
    """
    Your supervisor tasked you with **describing** a piece of text, specifically a prompt for a Large Language Model.
    Look at the input prompt and write and describe it **briefly** (with a few words or numerically) for each of the following categories:
    - length (how long is the prompt, shorter/longer than necessary ...)
    - structure (how does it start, where are the formatting brackets '{}' ...)
    - writing_style (is it in a dry style or is it interesting in some way?)
    - persona (imagine the person who could have written the prompt and describe them briefly)
    - theme (does the prompt try to put the reader into a specific setting, like some magical land etc.?)
    - identity (does the prompt identity assignment to the reader, like saying they are a proficient scientiest etc.?)
    - thinking_style (what style of thinking do the instructions promote in the reader?)
    - specificity (does the prompt go into specific step-by-step instructions or is it general?)
    - desired_outcome (what does the prompt ask for and in what way?)
    - examples_given  (does the prompt give any examples? what is their nature?)
    Respond with your characteristics for the categories (length, structure, writing style, persona, thinking style, specificity, desired outcome, examples given)
    """
    input_prompt: str = dspy.InputField()
    characteristics: dict[str,str] = dspy.OutputField()

class LamarckianDecoder(dspy.Signature):
    """
    Your supervisor tasked you with **generating a prompt** for a Large Language Model.
    Given several input/output examples, design a suitable zero-shot prompt for a Large Language Model for that task according to the provided characteristics.
    Here is the overview of the characteristics provided to specify the nature of the prompt:
    - length (how long is the prompt, shorter/longer than necessary ...)
    - structure (how does it start, where are the formatting brackets '{}' ...)
    - writing_style (is it in a dry style or is it interesting in some way?)
    - persona (imagine the person who could have written the prompt and describe them briefly)
    - theme (does the prompt try to put the reader into a specific setting, like some magical land etc.?)
    - identity (does the prompt identity assignment to the reader, like saying they are a proficient scientiest etc.?)
    - thinking_style (what style of thinking do the instructions promote in the reader?)
    - specificity (does the prompt go into specific step-by-step instructions or is it general?)
    - desired_outcome (what does the prompt ask for and in what way?)
    - examples_given  (does the prompt give any examples? what is their nature?)

    Having reflected on these characteristics, **design your prompt**. 
    Keep in mind to only **exactly** one pair of brackets '{}' in your prompts to indicate where the question should be inserted.
    """
    examples: str = dspy.InputField()
    #supervisor_hint: str = dspy.InputField()
    characteristics: dict[str,str] = dspy.InputField()
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