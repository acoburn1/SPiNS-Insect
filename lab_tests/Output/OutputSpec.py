from dataclasses import dataclass
from email.mime.image import MIMEImage
from enum import Enum
from typing import Optional, List


class PlotKind(str, Enum):
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"


class LineStyle(str, Enum):
    SOLID = "-"
    DASHED = "--"
    DOTTED = ":"
    DASHDOT = "-."


class Color(str, Enum):
    BLUE = "tab:blue"
    ORANGE = "tab:orange"
    GREEN = "tab:green"
    RED = "tab:red"
    PURPLE = "tab:purple"
    BROWN = "tab:brown"
    PINK = "tab:pink"
    GRAY = "tab:gray"
    OLIVE = "tab:olive"
    CYAN = "tab:cyan"
    BLACK = "black"


@dataclass
class Series:
    kind: PlotKind
    label: str
    x: List[float]
    y: List[float]
    yerr: Optional[List[float]] = None
    xerr: Optional[List[float]] = None
    ci_lower: Optional[List[float]] = None
    ci_upper: Optional[List[float]] = None
    color: Optional[Color] = None
    linestyle: Optional[LineStyle] = None


@dataclass
class OutputSpec:
    figure_id: str
    title: str
    x_label: str
    y_label: str
    series_list: Optional[List[Series]] = None
    matrix: Optional[List[List[float]]] = None