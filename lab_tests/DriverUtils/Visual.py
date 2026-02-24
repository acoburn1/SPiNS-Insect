import time
from dataclasses import dataclass
from tqdm import tqdm

# keep your colors as-is
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"

def _color_loss(loss):
    if loss < 0.01:
        return f"{GREEN}{loss:10.6f}{RESET}"
    elif loss < 0.1:
        return f"{YELLOW}{loss:10.6f}{RESET}"
    else:
        return f"{RED}{loss:10.6f}{RESET}"

def _bar(frac, width=18, color=GREEN):
    if frac < 0: frac = 0.0
    if frac > 1: frac = 1.0

    # thin style characters
    full_char = "━"      # thinner than █
    empty_char = "─"     # light line
    partials = ["", "╸", "╾", "━"]  # subtle partial fill

    total = frac * width
    full = int(total)
    rem = total - full

    # choose partial (4-step smoothness)
    p = int(rem * 4)

    if full >= width:
        full = width
        partial = ""
        empty = 0
    else:
        partial = partials[p]
        empty = width - full - (1 if partial else 0)

    return (
        f"{color}"
        + full_char * full
        + partial
        + RESET
        + DIM
        + empty_char * empty
        + RESET
    )

@dataclass
class VisualInfo:
    hls: int
    lr: float
    model_n: int
    epoch_n: int

    model_i: int = 0
    pair_t0: float = 0.0
    _line: tqdm | None = None

    def start_pair(self):
        self.pair_t0 = time.time()
        self._line = tqdm(
            total=1,
            position=0,
            leave=True,
            dynamic_ncols=True,
            bar_format="{desc}",
        )

    def close_pair(self):
        if self._line is not None:
            self._line.close()
            self._line = None

def next_model(vis: VisualInfo, model_i: int):
    vis.model_i = model_i

def progress_line(vis: VisualInfo, epoch_i: int, loss: float):
    if vis._line is None:
        return

    elapsed = time.time() - vis.pair_t0

    epoch_frac = (epoch_i + 1) / max(1, vis.epoch_n)
    overall_frac = (vis.model_i + epoch_frac) / max(1, vis.model_n)

    mw = len(str(vis.model_n))
    ew = len(str(vis.epoch_n))

    model_bar = _bar(overall_frac, width=18, color=CYAN)
    epoch_bar = _bar(epoch_frac, width=18, color=MAGENTA)

    line = (
        f"{CYAN}{BOLD}HLS={vis.hls:<3}{RESET} | "
        f"{CYAN}{BOLD}LR={vis.lr:<4}{RESET} | "
        f"{CYAN}{BOLD}MODELS {model_bar}{RESET} | "
        f"{DIM}t={elapsed:6.1f}s{RESET} | "
        f"{CYAN}{BOLD}M{vis.model_i+1:>{mw}}/{vis.model_n}{RESET} | "
        f"{MAGENTA}{BOLD}{epoch_bar}{RESET} | "
        f"{MAGENTA}E{epoch_i+1:>{ew}}/{vis.epoch_n}{RESET} | "
        f"loss={_color_loss(loss)}"
    )

    vis._line.set_description_str(line, refresh=True)

def progress_done(vis: VisualInfo):
    vis.close_pair()

def print_dim(s: str):
    print(f"{DIM}{s}{RESET}")

def print_cfgs(cfgs):
    def _print_dict(d, indent=2):
        if not d:
            return
        max_key_len = max(len(str(k)) for k in d.keys())

        for k, v in d.items():
            pad = " " * indent
            if isinstance(v, dict):
                print(f"{pad}{GREEN}{k:<{max_key_len}}{RESET} :")
                _print_dict(v, indent + 4)
            else:
                print(f"{pad}{GREEN}{k:<{max_key_len}}{RESET} : {YELLOW}{v}{RESET}")

    print(f"\n{CYAN}{BOLD}/ -------- configuration --------{RESET}")

    for subdir, cfg in cfgs.items():
        if cfg is None:
            continue
        print(f"\n{MAGENTA}{BOLD}[ {subdir} ]{RESET}")
        _print_dict(cfg, indent=2)

    print(f"\n{CYAN}{BOLD}\\ --------------------------------{RESET}\n")