from typing import Protocol, Iterable, List
from Output.schema.OutputSpec import OutputSpec

class Output(Protocol):
    name: str
    def generate_output(self, spec_cfg: dict, analysis_dir: str) -> list[OutputSpec]: ...