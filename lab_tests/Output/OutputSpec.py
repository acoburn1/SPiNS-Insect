# OutputSpec.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple


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


class YAxis(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Aspect(str, Enum):
    AUTO = "auto"
    EQUAL = "equal"


@dataclass
class RLine:
    val: float
    linestyle: LineStyle = LineStyle.DASHED
    color: Color = Color.GRAY
    alpha: float = 0.7
    linewidth: float = 1.5


@dataclass
class Series:
    kind: PlotKind
    label: str
    x: List[float]
    y: List[float]

    y_axis: YAxis = YAxis.LEFT

    yerr: Optional[List[float]] = None
    xerr: Optional[List[float]] = None
    ci_lower: Optional[List[float]] = None
    ci_upper: Optional[List[float]] = None

    color: Optional[Color] = None
    linestyle: Optional[LineStyle] = None

    marker: Optional[str] = "o"
    markersize: Optional[float] = 4.0
    linewidth: Optional[float] = 2.0
    alpha: Optional[float] = 1.0


@dataclass
class OutputSpec:
    figure_id: str
    title: str
    x_label: str
    y_label: str

    series_list: Optional[List[Series]] = None
    matrix: Optional[List[List[float]]] = None

    y2_label: Optional[str] = None

    x_lim: Optional[List[float]] = None
    y_lim: Optional[List[float]] = None
    y2_lim: Optional[List[float]] = None

    x_ref: Optional[List[RLine]] = None
    y_ref: Optional[List[RLine]] = None
    y2_ref: Optional[List[RLine]] = None

    x_ticks: Optional[List[float]] = None
    x_ticklabels: Optional[List[str]] = None
    y_ticks: Optional[List[float]] = None
    y_ticklabels: Optional[List[str]] = None

    grid: bool = True
    aspect: Aspect = Aspect.AUTO

    scatter_alpha: Optional[float] = 0.25
    scatter_marker: Optional[str] = None
    scatter_size: Optional[float] = None

    legend_fontsize: Optional[float] = 8
    legend_loc: str = "best"
    legend_ncol: Optional[int] = None

    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 300