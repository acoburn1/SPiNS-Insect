from typing import Protocol, Iterable, List
from dataclasses import dataclass
from Output.OutputSpec import OutputSpec
import yaml
from typing import Dict, List

class Evaluator(Protocol):
    name: str
    def run(self, cfg: dict, zarr_path: str, analysis_dir: str) -> list["OutputSpec"]: ...

class EpochEvaluator(Protocol):
    name: str
    def run(self, cfg: dict, zarr_path: str, analysis_dir: str, epoch: int) -> list["OutputSpec"]: ...
    
def build_registry(evaluators: List["Evaluator"]) -> Dict[str, "Evaluator"]:
    reg = {}
    for ev in evaluators:
        if ev.name in reg:
            raise ValueError(f"Duplicate evaluator name: {ev.name}")
        reg[ev.name] = ev
    return reg


def load_enabled_evaluators(yaml_path: str, registry: Dict[str, "Evaluator"]) -> List["Evaluator"]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    flags = cfg.get("evaluators", {})
    if not isinstance(flags, dict):
        raise ValueError("YAML field 'evaluators' must be a mapping of {name: true/false}.")

    enabled = []
    for name, on in flags.items():
        if not isinstance(on, bool):
            raise ValueError(f"Evaluator flag for '{name}' must be boolean, got {type(on).__name__}.")
        if not on:
            continue
        if name not in registry:
            raise ValueError(f"Unknown evaluator '{name}'. Available: {sorted(registry.keys())}")
        enabled.append(registry[name])

    return enabled


def run_enabled(zarr_path: str, save_path: str, enabled: List["Evaluator"]) -> List["OutputSpec"]:
    out: List["OutputSpec"] = []
    for ev in enabled:
        specs = ev.run(zarr_path, save_path)
        if isinstance(specs, list):
            out.extend(specs)
        else:
            out.append(specs)
    return out