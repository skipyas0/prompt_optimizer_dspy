import re
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
from my_signatures import Signature, Field

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
        with open("wtf.json", "w+") as f:
            json.dump(messages, f, indent=4)
        if temp is None:
            temp = self.temp
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            n=1,
        )
        return completion

    def predict(self, signature: Signature, temp=None, developer_prompt=None, max_tries=10, **kwargs):
        if temp is None:
            temp = self.temp
        if developer_prompt is None:
            developer_prompt = textwrap.dedent(
                """\
                Answer the request in JSON format according to the requirements.
                Input variables are specified in the 'inputs' field along with input *values*.
                Desired outputs are described in the 'outputs' field.
                Respond in JSON format to answer all output fields.
            """
            )
        # repeat attempt up to max_tries when response is invalid
        ret = None
        tries = 0
        while tries < max_tries and ret is None:
            inputs = sorted(signature.mandatory_inputs())
            print(f"Predict keys {sorted(list(kwargs.keys()))}, inputs {inputs}")
            assert sorted(list(kwargs.keys())) == inputs, f"Kwargs {kwargs} do not match the signature inputs {inputs}"
    
            signature_dict = signature.as_dict()
    
            for k, v in kwargs.items():
                signature_dict["inputs"][k]["value"] = v
            msgs = [
                {
                    "role": "developer",
                    "content": developer_prompt,
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
                ret_attempt = dict(json_obj)
                assert signature.matches_output(
                    ret_attempt
                ), f"LLM JSON response {ret} does not match the desired outputs {signature.mandatory_outputs()}"
                ret = ret_attempt
            except Exception as e:
                print(f"WARNING: JSON response parsing error for {content}: {str(e)}")
            tries += 1
        return ret
    
    def ask(self, question, temp=None):
        if temp is None:
            temp = self.temp
        sig = Signature.from_str(
            "question: str () -> answer: str (concise, clear and helpful answer to the user's question)"
        )
        response = self.predict(sig, question=question)
        return response["answer"]

    def chain_of_thought(
        self, signature: Signature, temp=None, **kwargs
    ) -> tuple[str, dict]:
        if temp is None:
            temp = self.temp
        signature = signature.copy()
        signature.update_outputs(
            [
                Field(
                    "reasoning",
                    str,
                    "Your internal thought process containing the rationale for your final answer.",
                )
            ]
        )

        ret = self.predict(signature, temp=temp, **kwargs)
        reasoning = ret.pop("reasoning")
        return reasoning, ret

    def chain_of_thought_sc(
        self, signature: Signature, temp=None, n_chains=5, **kwargs
    ):
        if temp is None:
            temp = self.temp
        logger.info(f"Running {n_chains} chains for CoT with self-consistency.")

        completions = [
            self.chain_of_thought(signature, temp=temp, **kwargs)
            for _ in range(n_chains)
        ]

        # majority voting
        answers = [re.sub(r"\s+", "", str(c[1]).lower()) for c in completions]
        counts = Counter(answers)
        most_common = counts.most_common(1)[0][0]
        index = answers.index(most_common)
        sc_answer = completions[index][1]
        reasonings = [c[0] for c in completions if c[1] == sc_answer]

        return reasonings, sc_answer

    def react(
        self,
        signature: Signature,
        temp=None,
        tools: list[Callable] = [],
        max_iters=10,
        **kwargs,
    ) -> tuple[list[dict], str]:
        if temp is None:
            temp = self.temp

        assert "finish" not in [
            tool.__name__ for tool in tools
        ], "The 'finish' tool is reserved for the system."

        signature.update_inputs(
            [
                Field(
                    "trajectory",
                    list,
                    "Your previous thoughts, tool uses (actions) and tool call results (observations).",
                ),
                Field(
                    "tools",
                    list,
                    "Tools that you can use to aide you in your thinking process.",
                ),
            ]
        )

        signature.update_outputs(
            [
                Field("tool_name", str, "Tool you decided to use in this step."),
                Field(
                    "tool_args",
                    list,
                    "All the arguments specified in the specified tool's signature as a Python list object.",
                ),
            ]
        )

        answer = None

        # add finish tool
        def finish(final_answer) -> None:
            """
            Use this tool to finish the response with the final answer.
            Answer in JSON format according to the requirements.
            Desired outputs are described in the 'outputs' field.
            Respond in JSON format to answer all output fields.
            """
            nonlocal answer
            answer = final_answer

        tools.append(finish)

        tool_info = [
            {
                "tool_name": tool.__name__,
                "tool_signature": str(inspect.signature(tool)),
                "tool_description": textwrap.dedent(tool.__doc__),
            }
            for tool in tools
        ]

        toolkit = {tool.__name__: tool for tool in tools}
        tool_name = None
        i = 0
        trajectory = []

        while answer is None and i < max_iters:
            kwargs.update({"trajectory": trajectory, "tools": tool_info})

            thought, content = self.chain_of_thought(signature, temp=temp, **kwargs)

            # parse reasoning and answer
            try:
                tool_name = content["tool_name"]
                tool_args = content["tool_args"]
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
                logger.warning(f"ReAct tool execution error: {str(e)}")
                tool_result = str(e)
                trajectory.append(
                    {
                        "role": "action",
                        "content": f"Your action resulted in an error:\n'{e}'\n Please double check tool names, arguments and format requirements and try again.",
                    }
                )

            i += 1
            logger.info(
                f"ReAct iteration {i}:\nThought: {thought}\nTool_name: {tool_name}\nTool_args: {tool_args}\nObservation: {tool_result}"
            )

        return trajectory, answer

    def program_of_thought(
        self, signature: Signature, temp=None, **kwargs
    ) -> tuple[str, str, str, str]:
        if temp is None:
            temp = self.temp

        code_signature = Signature(
            signature.input_fields,
            [Field(
                "code",
                str,
                "Python code that solves the user request. Only executable python code without external libraries usage.",
            )]
        )

        code_reasoning, code_ans = self.chain_of_thought(
            code_signature, temp=temp, **kwargs
        )
        code = code_ans["code"]
        code_result = utils.execute_code(code)

        signature.update_inputs(
            [
                Field(
                    "code_reasoning", str, "Your reasoning in the code generation step."
                ),
                Field("code", str, "Python code that solves the task."),
                Field("code_result", str, "Output of your code."),
            ]
        )
        kwargs.update(
            {"code_reasoning": code_reasoning, "code": code, "code_result": code_result}
        )

        answer_reasoning, answer = self.chain_of_thought(signature, temp=temp, **kwargs)

        return code_reasoning, code, answer_reasoning, answer

    def reflexion(
        self, signature: Signature, temp=None, iters=3, **kwargs
    ) -> tuple[list[dict], str]:
        if temp is None:
            temp = self.temp

        trajectory = []
        reasoning, answer = self.chain_of_thought(signature, temp=temp, **kwargs)
        trajectory.append(
            {"role": "solve", "content": {"reasoning": reasoning, "answer": answer}}
        )

        signature_with_feedback = signature.copy()
        signature_with_feedback.update_inputs(
            [
                Field("previous_reasoning", str, "Your reasoning in the last step."),
                Field("previous_answer", str, "Your answer in the last step."),
                Field("feedback", str, "Feedback you received for your last answer."),
            ]
        )

        eval_and_reflect_signature = Signature(
            [
                Field(
                    "original_request",
                    dict,
                    "Problem specification in an input/output format.",
                ),
                Field(
                    "student_reasoning", str, "Student's reasoning when solving the request."
                ),
                Field(
                    "student_answer",
                    dict,
                    "Student's answer to the problem in the specified format.",
                ),
            ],
            [
                Field(
                    "grade",
                    int,
                    "A grade 1-10 that reflects the correctness and completeness of the student's solution.",
                ),
                Field(
                    "feedback",
                    str,
                    "Feedback to the student on how to improve their solution.",
                ),
            ],
        )

        for i in range(iters):
            evaluation_reasoning, eval_and_reflect = self.chain_of_thought(
                eval_and_reflect_signature,
                temp=temp,
                original_request=signature.as_dict(),
                student_reasoning=reasoning,
                student_answer=answer
            )

            try:
                grade = eval_and_reflect["grade"]
                feedback = eval_and_reflect["feedback"]
            except Exception as e:
                logger.warning(f"Reflection parsing error: {e}")
                break

            trajectory.append(
                {
                    "role": "feedback",
                    "content": {
                        "reasoning": evaluation_reasoning,
                        "grade": grade,
                        "feedback": feedback,
                    },
                }
            )

            kwargs.update(
                {
                    "previous_reasoning": reasoning,
                    "previous_answer": answer,
                    "feedback": feedback,
                }
            )
            reasoning, answer = self.chain_of_thought(
                signature_with_feedback.copy(), temp=temp, **kwargs
            )

        return trajectory, answer

    def decompose(self, signature, temp=None, **kwargs) -> str:
        if temp is None:
            temp = self.temp
        signature_dict = signature.as_dict(prefix="original_")

        for k, v in kwargs.items():
            signature_dict["original_inputs"][k]["value"] = v
        decompose_signature = Signature.from_str(
            "original_signature: dict (Specification of the primary task for reference.) -> subtasks: list (Decomposition of the primary task into subtasks as a Python list of strings. The last subtask should complete the original task.)"
        )

        content = self.predict(decompose_signature, temp=temp, original_signature=signature_dict)
        print("decomp content",content)
        return content["subtasks"]

    def tree_of_thoughts(
        self,
        signature: Signature,
        temp=None,
        n_children=3,
        mode: Literal["dfs", "bfs"] = "bfs",
        bfs_top_k: int = 2,
        **kwargs,
    ) -> list[list[str]]:
        assert mode in ["dfs", "bfs"], "Invalid mode, choose 'dfs' or 'bfs'"
        if temp is None:
            temp = self.temp

        thought_steps = self.decompose(signature, temp=temp, **kwargs)
        #thought_steps += [
        #    "Combine the subtasks into a coherent response. This is the final step. Formulate your final answer to the original request."
        #]
        depth = len(thought_steps)

        logger.info(f"Problem was decomposed into {depth} subtasks:\n{thought_steps}")

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

        next_thought_signature = Signature(
            [
                Field(
                    "original_request",
                    dict,
                    "Original problem specification in an input/output format. This is for context only",
                ),
                Field(
                    "previous_thoughts",
                    list[tuple[str, str]],
                    "Your thoughts on the previous subtasks.",
                ),
                Field("current_subtask", str, "The subtask you are currently solving."),
            ],
            [
                Field(
                    "subtask_solution",
                    str,
                    "Systematic solution of the current subtask",
                ),
            ],
        )

        state_evaluation_signature = next_thought_signature.copy()
        state_evaluation_signature.update_inputs(
            [Field("current_subtask_solution", str, "Solution of the current subtask.")]
        )
        state_evaluation_signature.output_fields = [
            Field(
                "state_score",
                int,
                "Quality of the current subtask solution as an integer grade 0-100.",
            )
        ]
        signature_dict = signature.as_dict(prefix="original_")

        for k, v in kwargs.items():
            signature_dict["original_inputs"][k]["value"] = v

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
            elif mode == "bfs" and len(best_nodes) == 0:
                # check all tree depths before ending the search
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
            previous_thoughts = [
                (thought_steps[i], thought)
                for i, thought in enumerate(node.get_thought_chain())
            ]

            # thought step that will be solved by currently constructed node
            current_thought_step = thought_steps[node.gen]

            for _ in range(n_children):
                # create new thought
                next_thought = self.predict(
                    next_thought_signature,
                    temp=temp,
                    original_request=signature_dict,
                    previous_thoughts=previous_thoughts,
                    current_subtask=current_thought_step,
                )

                # score new thought
                evaluation = self.chain_of_thought(
                    state_evaluation_signature,
                    temp=temp,
                    original_request=signature_dict,
                    previous_thoughts=previous_thoughts,
                    current_subtask=current_thought_step,
                    current_subtask_solution=next_thought,
                )
                subtask_score = evaluation[1]["state_score"]

                subtask_score = utils.safe_int(subtask_score)
                next_step = node.gen + 1
                state_score = (score_so_far * node.gen + subtask_score) / next_step
                new_node = Node(next_thought, next_step, node, state_score)
                logger.info(
                    f"New thought of gen {next_step}: {next_thought}\nSubtask/State score: {subtask_score}/{state_score}"
                )
                node.children.append(new_node)
                nodes.append(new_node)
                if next_step == depth:
                    logger.info("Solution found!")
                    solution_nodes.append(new_node)

        # Get solution in the format specified by the original signature
        best_solution_node = sorted(solution_nodes, key=lambda n:n.score)[-1]
        signature.update_inputs([Field(
                    "previous_thoughts",
                    list[tuple[str, str]],
                    "Your thoughts on the previous subtasks.",
                )])
        kwargs.update({"previous_thoughts": [
                (thought_steps[i], thought)
                for i, thought in enumerate(best_solution_node.get_thought_chain())
            ]})
        answer = self.predict(signature, **kwargs)
        return [node.get_thought_chain() for node in solution_nodes], answer


model = ModelAPI()

if __name__ == "__main__":
    sig = Signature.from_str(
            "question: str () -> answer: int ()"
        )
    q = textwrap.dedent("""\
The proper divisors of 12 are 1, 2, 3, 4 and 6. A proper divisor of an integer $N$ is a positive divisor of $N$ that is less than $N$. What is the sum of the proper divisors of the sum of the proper divisors of 284?
                        """)
    def calculate(expression: str) -> float:
        """
        Evaluate the given mathematical expression
        """
        return eval(expression)
    #response = model.react(signature=sig, question=q, tools=[calculate])
    response = model.chain_of_thought_sc(signature=sig, question=q)
    print(response)

