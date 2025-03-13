from __future__ import annotations
import re

class Field:
    def __init__(self, name, type, desc):
        self.name = name
        self.type = type
        self.desc = desc


class Signature:
    def __init__(self, input_fields: list[Field], output_fields: list[Field]):
        self.input_fields = input_fields
        self.output_fields = output_fields

    def mandatory_inputs(self) -> list[str]:
        return [f.name for f in self.input_fields]

    def mandatory_outputs(self) -> list[str]:
        return [f.name for f in self.output_fields]
    
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
                    name, type, desc = match.groups()
                    fields.append(Field(name, type, desc))
                else:
                    raise ValueError(f"Wrong signature format in '{string_signature}'")
        return Signature(io_fields[0], io_fields[1])

    def as_dict(self) -> dict:
        return {
            "inputs": {
                f"{f.name}": {"type": f.type, "description": f.desc, "value": None}
                 for f in self.input_fields
            },
            "outputs": {
                f"{f.name}": {"type": f.type, "description": f.desc,}
                for f in self.output_fields
            } 
        }

ask = Signature.from_str("question: str () -> answer: str (concise, clear and helpful answer to the user's question)")
chain_of_thought = lambda answer_type: Signature.from_str(f"question: str () -> reasoning: str (step-by-step thinking process that leads to the answer);; answer: {answer_type} (final answer)")


if __name__ == "__main__":
    import json
    s = Signature.from_str(
        "question: str (question by the user);; context: str (context relevant to the question);; number_of_paragraphs: int (in how many paragraphs to answer) -> reasoning: str ();; answer: str ();; confidence: float (how confident you are in your answer from 0.0 to 1.0)"
    )
    sig = s.as_dict()
    with open("sig.json", "w+") as f:
        json.dump(sig, f, indent=4)
    print(s.mandatory())