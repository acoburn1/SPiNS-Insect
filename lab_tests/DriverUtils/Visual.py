import time
from dataclasses import dataclass
from tqdm import tqdm

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
    if frac < 0:
        frac = 0.0
    if frac > 1:
        frac = 1.0

    full_char = "━"
    empty_char = "─"
    partials = ["", "╸", "╾", "━"]

    total = frac * width
    full = int(total)
    rem = total - full
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


def print_dim(s: str):
    print(f"{DIM}{s}{RESET}")


@dataclass
class ModelVisualInfo:
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

    def next_model(self, model_i: int):
        self.model_i = model_i

    def progress_line(self, epoch_i: int, loss: float):
        if self._line is None:
            return

        elapsed = time.time() - self.pair_t0

        epoch_frac = (epoch_i + 1) / max(1, self.epoch_n)
        overall_frac = (self.model_i + epoch_frac) / max(1, self.model_n)

        mw = len(str(self.model_n))
        ew = len(str(self.epoch_n))

        model_bar = _bar(overall_frac, width=18, color=CYAN)
        epoch_bar = _bar(epoch_frac, width=18, color=MAGENTA)

        line = (
            f"{CYAN}{BOLD}HLS={self.hls:<3}{RESET} | "
            f"{CYAN}{BOLD}LR={self.lr:<4}{RESET} | "
            f"{CYAN}{BOLD}MODELS {model_bar}{RESET} | "
            f"{DIM}t={elapsed:6.1f}s{RESET} | "
            f"{CYAN}{BOLD}M{self.model_i+1:>{mw}}/{self.model_n}{RESET} | "
            f"{MAGENTA}{BOLD}{epoch_bar}{RESET} | "
            f"{MAGENTA}E{epoch_i+1:>{ew}}/{self.epoch_n}{RESET} | "
            f"loss={_color_loss(loss)}"
        )

        self._line.set_description_str(line, refresh=True)

    def progress_done(self):
        self.close_pair()

    @staticmethod
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


@dataclass
class EvalVisualInfo:
    hls: int
    lr: float
    pair_i: int
    pair_n: int
    eval_n: int
    model_n: int
    epoch_n: int

    t0: float = 0.0
    _line: tqdm | None = None
    eval_name: str = ""
    eval_i: int = 0
    _note: str = ""

    def start(self):
        self.t0 = time.time()
        self._line = tqdm(
            total=1,
            position=0,
            leave=True,
            dynamic_ncols=True,
            bar_format="{desc}",
        )

    def close(self):
        if self._line is not None:
            self._line.close()
            self._line = None

    def set_eval(self, eval_name: str, eval_i: int):
        self.eval_name = str(eval_name)
        self.eval_i = int(eval_i)
        self._note = ""
        self._render(grid_frac=None, me_frac=None)

    def note(self, s: str):
        self._note = str(s)
        self._render(grid_frac=None, me_frac=None)

    def fast_done(self):
        self._render(grid_frac=(self.pair_i * self.eval_n + (self.eval_i + 1)) / max(1, self.pair_n * self.eval_n), me_frac=1.0)

    def update(self, model_i: int, epoch_i: int):
        if self._line is None:
            return

        me_done = int(model_i) * self.epoch_n + (int(epoch_i) + 1)
        me_total = max(1, self.model_n * self.epoch_n)
        me_frac = me_done / me_total

        grand_done = self.pair_i * self.eval_n + self.eval_i + me_frac
        grand_total = max(1, self.pair_n * self.eval_n)
        grid_frac = grand_done / grand_total

        self._render(grid_frac=grid_frac, me_frac=me_frac)

    def _render(self, *, grid_frac, me_frac):
        if self._line is None:
            return

        elapsed = time.time() - self.t0

        grid_bar = _bar(grid_frac if grid_frac is not None else 0.0, width=18, color=CYAN)
        me_bar = _bar(me_frac if me_frac is not None else 0.0, width=18, color=MAGENTA)

        note = f" | {DIM}{self._note}{RESET}" if self._note else ""

        desc = (
            f"{CYAN}{BOLD}HLS={self.hls:<3}{RESET} | "
            f"{CYAN}{BOLD}LR={self.lr:<6}{RESET} | "
            f"{CYAN}{BOLD}EVAL={self.eval_name:<14}{RESET} | "
            f"{CYAN}{BOLD}GRID {grid_bar}{RESET} | "
            f"{MAGENTA}{BOLD}M×E {me_bar}{RESET} | "
            f"{DIM}t={elapsed:6.1f}s{RESET}"
            f"{note}"
        )

        self._line.set_description_str(desc, refresh=True)