from dotenv import load_dotenv
import openai
import os
import json
import textwrap
from typing import Callable, Literal
import inspect
import logging
from collections import Counter
import utils
from my_signatures import Signature

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
base = os.getenv("BASE_PATH") or "."
handler = logging.FileHandler(f"{base}/model_api.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class ModelAPI:
    def __init__(self, temp=0.6, api_key=None, identifier=None, logging=True) -> None:
        self.temp = temp
        self.identifier = identifier
        load_dotenv()
        port = os.getenv("VLLM_MY_PORT")
        api_key = api_key if api_key else os.getenv("API_KEY_UNIVERSAL")
        self.logging = logging
        if port:
            # hosted locally
            self.model = ""
            self.client = openai.OpenAI(
                base_url=f"http://localhost:{port}/v1",
                api_key="EMPTY",
            )
        elif api_key:
            # hosted on openai
            self.model = "gpt-4o-mini"
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError("No API key or port specified")
        logger.info(
            f"ModelAPI initialized with temp={temp}, api_key={api_key}, identifier={identifier}, logging={logging}"
        )

    def forward(self, messages, temp=None):
        if temp is None:
            temp = self.temp
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
        )
        return completion

    def predict(self, signature: Signature, temp=None, **kwargs):
        if temp is None:
            temp = self.temp
        assert sorted(list(kwargs.keys())) == sorted(
            signature.mandatory_inputs()
        ), f"Kwargs {kwargs} do not match the signature inputs {signature.input_fields}"

        signature_dict = signature.as_dict()

        for k, v in kwargs.items():
            signature_dict["inputs"][k]["value"] = v

        msgs = [
            {
                "role": "system",
                "content": textwrap.dedent(
                    """\
                Answer the request in a JSON format according to the requirements.
                Input variables are specified in the 'inputs' field along with input *values*.
                Pay special attention to additional *user instructions or questions* which might appear in the 'inputs' field.
                Desired outputs are described in the 'outputs' field.
                Respond in a JSON format to answer all output fields.
                Example of your output:
                {
                    "output1": "value of the first 'str' output",
                    "output2": 1,
                    "output3": "last output was of type 'int' and this is a 'str'"
                }
            """
                ),
            },
            {"role": "user", "content": json.dumps(signature_dict)},
        ]

        try:
            completion = self.forward(messages=msgs, temp=temp)
        except openai.BadRequestError as e:
            print(f"WARNING: openai.BadRequestError for: {msgs}")
            completion = {
                "error": "openai.BadRequestError",
                "choices": [{"message": {"content": str(e)}}],
            }

        content = completion.choices[0].message.content
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
            json_str = json_str.replace("\n", "").strip()
            json_str = json_str.replace("\\", "\\\\")  # Escape backslashes
            json_obj = json.loads(json_str)
            ret = dict(json_obj)
            assert sorted(list(ret.keys())) == sorted(
                signature.mandatory_outputs()
            ), f"LLM JSON response {ret} does not match the desired outputs {signature.output_fields}"
            return ret
        except Exception as e:
            print(f"WARNING: JSON response parsing error for {content}: {str(e)}")
            return None

    def ask(self, question, temp=None):
        if temp is None:
            temp = self.temp
        sig = Signature.from_str(
            "question: str () -> answer: str (concise, clear and helpful answer to the user's question)"
        )
        response = self.predict(sig, question=question)
        return response["answer"]

    def chain_of_thought(self, question, temp=None) -> tuple[str, str]:
        if temp is None:
            temp = self.temp

        # cot prompt template
        question = textwrap.dedent(
            f"""\
        The user has the following request: 
        '{question}'
        Use systematic step-by-step thinking to provide a clear, organized, and thoughtful response.
        Answer in JSON format with "reasoning" and "answer" fields.
        Fill the "reasoning" field with your internal thought process and the "answer" field with the final answer only.
        For your response, refer to the following example:
        Example response:
        {{
        'reasoning': 'I will first analyze the problem and then provide a solution.',
        'answer': 'The answer to the request.'
        }}
        The 'answer' field should be in the format specified by the user.
        """
        )

        content = self.predict(question, temp=temp)
        # print("COT Content:", content)
        # parse reasoning and answer
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
            json_str = json_str.replace("\n", "").strip()
            json_str = json_str.replace("\\", "\\\\")  # Escape backslashes
            json_obj = json.loads(json_str)
            reasoning = json_obj["reasoning"]
            answer = json_obj["answer"]
        except Exception as e:
            logger.warning(f"CoT json parsing error: {e}")
            reasoning = ""
            answer = content
        logger.info(f"CoT reasoning: {reasoning}\nCoT answer: {answer}")
        return reasoning, answer

    def chain_of_thought_sc(
        self, question, temp=None, n_chains=5
    ) -> tuple[list[str], str]:
        if temp is None:
            temp = self.temp
        logger.info(f"Running {n_chains} chains for CoT with self-consistency.")

        completions = [
            self.chain_of_thought(question, temp=temp) for _ in range(n_chains)
        ]

        # majority voting
        counts = Counter([answer for _, answer in completions])
        sc_answer = counts.most_common(1)[0][0]
        logger.info(f"CoTSC answer: {sc_answer}")
        reasonings = [
            reasoning for reasoning, answer in completions if answer == sc_answer
        ]

        return reasonings, sc_answer

    def react(
        self, question, temp=None, tools: list[Callable] = [], max_iters=10
    ) -> tuple[list[dict], str]:
        if temp is None:
            temp = self.temp

        assert "finish" not in [
            tool.__name__ for tool in tools
        ], "The 'finish' tool is reserved for the system."

        answer = None

        # add finish tool
        def finish(final_answer: str) -> None:
            """
            Use this tool to finish the response with the final answer.
            """
            nonlocal answer
            answer = final_answer

        tools.append(finish)

        tool_info = "\n".join(
            [
                f"Tool name: {tool.__name__}\nTool signature: {str(inspect.signature(tool))}\nTool description: {textwrap.dedent(tool.__doc__)}\n"
                for tool in tools
            ]
        )

        toolkit = {tool.__name__: tool for tool in tools}
        tool_name = None
        i = 0
        trajectory = []

        while tool_name != "finish" and i < max_iters:
            history_formatted = "\n".join(
                [
                    f"Role: {msg['role']}\nContent: {msg['content']}\n"
                    for msg in trajectory
                ]
            )
            history_text_optional = (
                f"History:\n{history_formatted}" if history_formatted else ""
            )
            # react prompt template
            question = textwrap.dedent(
                f"""\
            The user has asked you a question: 
            '{question}'
            Use systematic step-by-step thinking to provide a clear, organized, and thoughtful response.
            In your thinking process, you are aided by *tools*.
            Tools:
            {tool_info}
            Each tool use results in an *observation* that will be added to the history.
            {history_text_optional}
            Answer in JSON format with "thought", "tool_name" and "tool_args" fields.
            Fill the "thought" field with your internal thought process, the "tool_name" field with the tool name, and the "tool_args" field with the tool arguments in a list format.
            In case you want to finish the response, use the "finish" tool with the final answer as the argument.
            """
            )
            content = self.predict(question, temp=temp)

            # parse reasoning and answer
            try:
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
                json_str = json_str.replace("\n", "").strip()
                json_str = json_str.replace("\\", "\\\\")  # Escape backslashes
                json_obj = json.loads(json_str)
                thought = json_obj["thought"]
                tool_name = json_obj["tool_name"]
                tool_args = json_obj["tool_args"]
            except Exception as e:
                logger.warning(f"ReAct json parsing error: {e}")
                answer = content
                break

            # execute tool and update history
            trajectory.append({"role": "thought", "content": f"{thought}"})
            trajectory.append(
                {
                    "role": "action",
                    "content": {"tool_name": tool_name, "tool_args": tool_args},
                }
            )
            try:
                tool = toolkit[tool_name]
                tool_result = tool(*tool_args)
                trajectory.append({"role": "observation", "content": f"{tool_result}"})
            except Exception as e:
                logger.warning(f"ReAct tool execution error: {e}")
                trajectory.append(
                    {
                        "role": "action",
                        "content": "Your action resulted in an error. Please double check tool names, arguments and format requirements and try again.",
                    }
                )

            i += 1
            logger.info(
                f"ReAct iteration {i}:\nThought: {thought}\nTool_name: {tool_name}\nTool_args: {tool_args}\nObservation: {tool_result}"
            )

        return trajectory, answer

    def program_of_thought(self, question, temp=None) -> tuple[str, str, str, str]:
        if temp is None:
            temp = self.temp

        # cot prompt template
        code_prompt = textwrap.dedent(
            f"""\
        Generate Python code that solves the following problem:
        "{question}"
        Your final answer should contain only executable Python code.
        Do not use external libraries.
        Do not try to run the code or answer the problem directly, just generate the Python code.
        """
        )

        logger.info(f"PoT program synthesis")
        code_reasoning, code = self.chain_of_thought(code_prompt, temp=temp)

        code_result = utils.execute_code(code)

        logger.info(f"PoT code execution result: {code_result}")
        answer_prompt = textwrap.dedent(
            f"""\
        {question}
        In previous steps, you have generated the following Python code:
        {code}
        with reasoning:
        {code_reasoning}
        Output of your code is:
        {code_result}
        """
        )

        logger.info(f"PoT final answer")
        answer_reasoning, answer = self.chain_of_thought(answer_prompt, temp=temp)

        return code_reasoning, code, answer_reasoning, answer

    def reflexion(self, question, temp=None, iters=3) -> tuple[list[dict], str]:
        if temp is None:
            temp = self.temp

        trajectory = []
        reasoning, answer, reflection = None, None, None

        for i in range(iters + 1):

            # solve attempt
            if reasoning is None or answer is None or reflection is None:
                solve_prompt = question
            else:
                solve_prompt = textwrap.dedent(
                    f"""\
                {question}
                In a previous step, you answered in the following way:
                Reasoning:
                {reasoning}
                Answer:
                {answer}
                You received the following feedback:
                {reflection}
                Improve your solution based on the feedback and try again.
                """
                )
            logger.info(f"Reflexion iteration {i}: SOLVE")
            reasoning, answer = self.chain_of_thought(solve_prompt, temp=temp)
            trajectory.append(
                {"role": "solve", "content": {"reasoning": reasoning, "answer": answer}}
            )

            # no eval/reflect step on final iteration
            if i == iters:
                break

            # evaluate solution
            evaluation_prompt = textwrap.dedent(
                f"""\
            You are a teacher who gave the following problem to a student:
            '{question}'
            The student has provided the following solution:  
            Reasoning:
            '{reasoning}'
            Answer:
            '{answer}'
            Evaluate the student's solution and grade it.
            Aspects to consider in your evaluation:
            - Correctness
            - Clarity
            - Completeness
            - Conciseness
            - Efficiency
            Your final answer should consist only of an integer (grade) 1-10 that reflects the solution quality and accuracy.
            """
            )
            logger.info(f"Reflexion iteration {i}: EVALUATION")
            evaluation_reasoning, evaluation = self.chain_of_thought(
                evaluation_prompt, temp=temp
            )
            trajectory.append(
                {
                    "role": "evaluation",
                    "content": {
                        "reasoning": evaluation_reasoning,
                        "answer": evaluation,
                    },
                }
            )

            # reflection

            reflection_prompt = textwrap.dedent(
                f"""\
            Evaluate the solution to the following problem:
            {question}
            The user has provided the following solution:  
            Reasoning:
            {reasoning}
            Answer:
            {answer}
            The solution was given a score {evaluation} with reasoning:
            {evaluation_reasoning}
            Provide feedback on the solution and suggest improvements.
            """
            )
            logger.info(f"Reflexion iteration {i}: REFLECTION")
            reflection = self.predict(reflection_prompt, temp=temp)
            logger.info(reflection)
            trajectory.append({"role": "reflection", "content": {"answer": reflection}})

        return trajectory, answer

    def decompose(self, question, temp=None, min_subtasks=3, max_subtasks=10) -> str:
        if temp is None:
            temp = self.temp
        prompt = textwrap.dedent(
            f"""\
        For the following request, generate a list of subtasks that need to be completed in order to fulfill the request.
        Request: 
        '{question}'
        Decompose the problem into several small problems to be tackled sequentially.
        Aim for {min_subtasks}-{max_subtasks} clearly defined subtasks.
        Respond with the subtasks only without any additional information.
        Separate each subtask with a new line.
        """
        )
        return self.predict(prompt, temp=temp)

    def tree_of_thoughts(
        self,
        question,
        temp=None,
        min_depth=6,
        max_depth=8,
        n_children=3,
        mode: Literal["dfs", "bfs"] = "bfs",
        bfs_top_k: int = 2,
    ) -> list[list[str]]:
        assert mode in ["dfs", "bfs"], "Invalid mode, choose 'dfs' or 'bfs"
        if temp is None:
            temp = self.temp

        thought_steps_str = self.decompose(
            question, temp=temp, min_subtasks=min_depth, max_subtasks=max_depth
        )
        thought_steps = [
            step for step in thought_steps_str.split("\n") if len(step) > 0
        ]
        thought_steps += [
            "Combine the subtasks into a coherent response. This is the final step. Formulate your final answer to the original request."
        ]
        depth = len(thought_steps)
        logger.info(
            f"Problem {question} was decomposed into {depth} subtasks:\n{thought_steps_str}"
        )

        class Node:
            def __init__(self, thought, gen, parent, score):
                self.thought = thought
                self.gen = gen
                self.parent = parent
                self.score = score
                self.children = []
                self.visited = False

            def get_thought_chain(self):
                chain = []
                node = self
                while node.parent != None:
                    chain.append(node.thought)
                    node = node.parent
                return chain[::-1]

        next_thought_prompt = textwrap.dedent(
            """\
        Your task is to perform one step that will bring you closer to solving the user's request
        Request:
        '{question}'
        {previous_thoughts}
        Solve this subtask and provide a clear, organized, and thoughtful response.
        {thought_step}                    
        """
        )
        previous_thoughts_intro = "Previous thought steps:\n"

        nodes = [Node("", 0, None, 0)]
        solution_nodes = []
        current_step = 0
        best_nodes = []
        while True:
            # node selection
            if mode == "dfs":
                # simply find the best performing node that has not been visited yet
                best_nodes = sorted(
                    filter(lambda node: not node.visited, nodes),
                    key=lambda node: node.score,
                    reverse=True,
                )
            elif mode == "bfs":
                # if
                if len(best_nodes) == 0:
                    # check all steps before ending the search
                    while current_step < depth:
                        # get best nodes from this step
                        not_visited_in_step = filter(
                            lambda node: not node.visited and node.gen == current_step,
                            nodes,
                        )
                        best_nodes_candidates = sorted(
                            not_visited_in_step,
                            key=lambda node: node.score,
                            reverse=True,
                        )

                        # only use the best k nodes and mark others as visited
                        for i, node in enumerate(best_nodes_candidates):
                            if i < bfs_top_k:
                                best_nodes.append(node)
                            else:
                                node.visited = True
                        # if something was found, finish at this step
                        if len(best_nodes) > 0:
                            break
                        # otherwise, move to the next step
                        current_step += 1

            if len(best_nodes) == 0:
                logger.info("No more nodes to visit.")
                break

            node = best_nodes.pop(0)
            score_so_far = node.score
            node.visited = True
            # format thought chain summary
            previous_thoughts = node.get_thought_chain()
            previous_thoughts_str = ""
            for i, thought in enumerate(previous_thoughts):
                previous_thoughts_str += (
                    f"Subtask {i+1}: {thought_steps[i]}\nSubtask solution:\n{thought}\n"
                )
            if len(previous_thoughts_str) > 0:
                previous_thoughts_str = previous_thoughts_intro + previous_thoughts_str

            # thought step that will be solved by currently constructed node
            current_thought_step = thought_steps[node.gen]
            prompt = next_thought_prompt.format(
                question=question,
                previous_thoughts=previous_thoughts_str,
                thought_step=current_thought_step,
            )

            for _ in range(n_children):
                # create new thought
                next_thought = self.predict(prompt, temp=temp)
                state_evaluation_prompt = textwrap.dedent(
                    f"""\
                Evaluate the progress made on solving the user's request.
                Request:
                '{question}'
                {previous_thoughts_str}
                The current subtask is:
                '{current_thought_step}'
                The current subtask has been solved by the following thought step.
                '{next_thought}'
                Now evaluate the current subtask solution.
                Areas to consider in your evaluation:
                - Correctness: Is the current subtask solved correctly?
                - Clarity: Is the solution clear and easy to understand?
                - Factuality: Is the solution based on facts and evidence?
                - Logic: Is the solution logically sound?
                - Quality: Think about the solution quality, including language, structure, and coherence.
                Rate the solution for each criteria and then combine the scores into a final evaluation.
                Only consider the current subtask solution in your evaluation independently of previous thought steps.
                Your final answer should be an integer state score from 0 to 100, where 0 means solution is irrelevant to the subtask and 100 means that the subtask is solved perfectly.                                                                
                """
                )

                # score new thought
                subtask_score = self.chain_of_thought(
                    state_evaluation_prompt, temp=temp
                )[1]
                subtask_score = utils.safe_int(subtask_score)
                next_step = node.gen + 1
                state_score = score_so_far + subtask_score
                new_node = Node(next_thought, next_step, node, state_score)
                logger.info(
                    f"New thought of gen {next_step}: {next_thought}\nSubtask/State score: {subtask_score}/{state_score}"
                )
                node.children.append(new_node)
                nodes.append(new_node)
                if next_step == depth:
                    logger.info("Solution found!")
                    solution_nodes.append(new_node)

        return [node.get_thought_chain() for node in solution_nodes]


model = ModelAPI()