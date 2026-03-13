import yaml
from pathlib import Path

def _get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def _resolve_cfg(cli_value: str | None, base_dir: str) -> dict:
    if not cli_value:
        return f"{base_dir}/default.yaml"
    p = Path(cli_value)
    if p.suffix == "":
        p = p.with_suffix(".yaml")
    s = str(p)
    if not p.is_absolute() and ("/" not in s and "\\" not in s):
        p = Path(base_dir) / p.name
    return p

def resolve_cfgs(args, idv: bool=False, base_dir: str="configs"):
    cfgs = {}
    for arg_name, value in vars(args).items():
        if not arg_name.endswith("_config"):
            continue
        subdir = arg_name.replace("_config", "")
        default_dir = Path(base_dir) / subdir
        cfgs[subdir] = _get_config(_resolve_cfg(value, str(default_dir)))
    
    return cfgs if not idv else (cfgs.get("data"), cfgs.get("model"), cfgs.get("output"), cfgs.get("probe"), cfgs.get("directory"))

def get_stub(args, base_dir: str="configs"):
    stub = ""
    for arg_name, value in vars(args).items():
        if not arg_name.endswith("_config") or arg_name.startswith("output") or arg_name.startswith("directory"):
            continue
        start = arg_name[0]
        subdir = arg_name.replace("_config", "")
        default_dir = Path(base_dir) / subdir
        stub += f"{start}__{Path(_resolve_cfg(value, str(default_dir))).stem}&"
    stub = stub.rstrip("&")
    return stub