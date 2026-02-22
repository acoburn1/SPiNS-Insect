import yaml
from pathlib import Path

def _get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def _resolve_cfg(cli_value: str | None, base_dir: str) -> str:
    if not cli_value:
        return _get_config(f"{base_dir}/default.yaml")
    p = Path(cli_value)
    if p.suffix == "":
        p = p.with_suffix(".yaml")
    s = str(p)
    if not p.is_absolute() and ("/" not in s and "\\" not in s):
        p = Path(base_dir) / p.name
    return _get_config(p)

def resolve_cfgs(args, idv: bool=False, base_dir: str = "configs"):
    cfgs = {}
    for arg_name, value in vars(args).items():
        if not arg_name.endswith("_config"):
            continue
        subdir = arg_name.replace("_config", "")
        default_dir = Path(base_dir) / subdir
        cfgs[subdir] = _resolve_cfg(value, str(default_dir))
    
    return cfgs if not idv else (cfgs.get("data"), cfgs.get("model"), cfgs.get("output"), cfgs.get("probe"))
    

def print_cfgs(cfgs):
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"

    print(f"\n{CYAN}{BOLD}/ -------- configuration -------- \\{RESET}")

    for subdir, cfg in cfgs.items():
        if cfg is None:
            continue
        print(f"\n{MAGENTA}{BOLD}[ {subdir} ]{RESET}")
        max_key_len = max(len(k) for k in cfg.keys())
        for k, v in cfg.items():
            print(f"  {GREEN}{k:<{max_key_len}}{RESET} : {YELLOW}{v}{RESET}")
    print(f"\n{CYAN}{BOLD}\\ -------------------------------- /{RESET}\n")
