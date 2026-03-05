from dataclasses import dataclass, field
from Eval.Protocol import Evaluator
from Output import EpochsHLSwK95Heatmap
from Output.Protocol import Output
from Eval.PCA import K95Evaluator
from Eval.Correlation import CorrelationEvaluator, MatrixCorrelationEvaluator, LossEvaluator
from Eval.RatioExemplar import RatioTestEvaluator
from Eval.SigE import SignificantEpochEvaluator
from Output.SeriesCorrelation import SeriesCorrelationOutput
from Output.MatrixCorrelation import MatrixCorrelationOutput
from Output.K95HLS import K95HLSOutput
from Output.CorrelationHLS import CorrelationHLSOutput
from Output.EpochsHLSwK95Heatmap import EpochsHLSwK95HeatmapOutput
from Output.SigEHLS import SigEHLSOutput
from Output.WithinVsBetweenCorrelation import WithinVsBetweenCorrelationOutput

@dataclass
class Dep:
        evaluators: list[object]
        output_function: object
        hyperd: bool = False

@dataclass
class DependenciesObject:
    cfgs: dict[Output, dict] = field(default_factory=dict)
    evaluation_fns: list[Evaluator] = field(default_factory=list)
    output_fns: list[Output] = field(default_factory=list)
    hyperd_output_fns: list[Output] = field(default_factory=list)
    sige: SignificantEpochEvaluator = None

dependencies = {
    "SeriesCorrelation": Dep([CorrelationEvaluator(), LossEvaluator()], SeriesCorrelationOutput()),
    "MatrixCorrelation": Dep([MatrixCorrelationEvaluator()], MatrixCorrelationOutput()),
    "RatioOverEpochs": Dep([RatioTestEvaluator()], None),
    "SCurve": Dep([RatioTestEvaluator()], None),
    "K95Bars": Dep([K95Evaluator()], None),
    "K95OverEpochs": Dep([K95Evaluator()], None),
    "WithinVsBetweenCorrelation": Dep([MatrixCorrelationEvaluator()], WithinVsBetweenCorrelationOutput()),
    "K95-HLS": Dep([K95Evaluator()], K95HLSOutput(), hyperd=True),
    "Correlation-HLS": Dep([CorrelationEvaluator()], CorrelationHLSOutput(), hyperd=True),
    "Epochs-HLSwK95Heatmap": Dep([K95Evaluator()], EpochsHLSwK95HeatmapOutput(), hyperd=True),
    "SigE-HLS": Dep([CorrelationEvaluator()], SigEHLSOutput(), hyperd=True)
    }


def get_dependencies(o_cfg):
    """ 
    Given an output config file, returns the list of necessary evaluators and output functions, followed by sige.
    sige will be SignificantEpochEvaluator() if any output requires it, otherwise None.
    """

    d_obj = DependenciesObject()

    for key, subcfg in o_cfg.items():
        if subcfg.get("present", False):
            dep = dependencies.get(key)
            evs = dep.evaluators
            out = dep.output_function
            for ev in evs:
                if ev is not None and not any(isinstance(existing, type(ev)) for existing in d_obj.evaluation_fns):
                    d_obj.evaluation_fns.append(ev)
            if out is not None:
               d_obj.cfgs.update({out: subcfg})
               d_obj.output_fns.append(out) if not dep.hyperd else d_obj.hyperd_output_fns.append(out)
        if d_obj.sige is None and subcfg.get("epochs", False) == 'sig':
            d_obj.sige = SignificantEpochEvaluator()

    return d_obj