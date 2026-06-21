from typing import Protocol, Iterable, List
from Output.schema.OutputSpec import OutputSpec

class Output(Protocol):
    name: str
    hyperd: bool
    def generate_output(self, sub_cfg: dict, analysis_dir: str) -> list[OutputSpec]: ...