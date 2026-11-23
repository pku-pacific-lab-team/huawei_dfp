"""Shape-only Decode embedding lookup; full table capacity, selected-row traffic."""
from dataclasses import dataclass

from .blocks import checked_inputs
from .types import OperatorSpec, TensorSpec, output_name as select_output_name


@dataclass(frozen=True)
class Embedding:
    indices: TensorSpec
    weight: TensorSpec
    output_name: str | None = None
    operation = "embedding"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if self.indices.dtype != "INT32" or len(self.indices.shape) != 1:
            raise ValueError("Decode Embedding requires INT32 indices [batch], one token per batch item")
        if self.weight.dtype != "INT8" or len(self.weight.shape) != 2:
            raise ValueError("Embedding weight must be INT8 [vocabulary, hidden]")
        if self.weight.shape[0] > 2**31:
            raise ValueError("Vocabulary row indices must fit nonnegative signed INT32")
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.indices, self.weight)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs,
                            (TensorSpec(self.output_name, (self.indices.shape[0], self.weight.shape[1]), "INT8"),),
                            attributes={"index_values": "symbolic_assumed_in_range", "duplicate_row_reuse": False,
                                        "table_residency": "complete_table", "address_instruction_model": False})
