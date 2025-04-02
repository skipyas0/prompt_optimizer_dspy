from __future__ import annotations
import re
import utils

class Field:
    def __init__(self, name: str, type: type, desc: str):
        self.name = name
        self.type = type
        self.desc = desc


class Signature:
    def __init__(self, input_fields: list[Field], output_fields: list[Field], instructions: str = ""):
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
        instructions = {prefix+"instructions": self.instructions} if len(self.instructions) > 0 else {}
        # Dict join operator
        return instructions | {
            prefix+"inputs": {
                f"{f.name}": {"type": str(f.type), "description": f.desc, "value": None}
                 for f in self.input_fields
            },
            prefix+"outputs": {
                f"{f.name}": {"type": str(f.type), "description": f.desc,}
                for f in self.output_fields
            } 
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
    

