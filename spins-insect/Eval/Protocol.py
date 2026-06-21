from typing import Protocol, Iterable, List
from dataclasses import dataclass
import numpy as np
import yaml
from typing import Dict, List
from DriverUtils.Visual import EvalVisualInfo

class Evaluator(Protocol):
    name: str
    def run(self, cfg, zarr_path: str, vis: EvalVisualInfo=None) -> tuple[np.ndarray, dict]: ...