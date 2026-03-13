from dataclasses import dataclass, field

from Eval.Protocol import Evaluator
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
from Output.RatioOverEpochs import RatioOverEpochsOutput
from Output.SCurve import SCurveOutput
from Output.AllModels33OverEpochs import AllModels33OverEpochsOutput
from Output.AllModelsSCurve import AllModelsSCurveOutput

@dataclass
class Dep:
    evaluators: list[type[Evaluator]]
    output_function: type[Output] | None
    hyperd: bool = False


@dataclass
class DependenciesObject:
    cfgs: dict[Output, dict] = field(default_factory=dict)
    evaluation_fns: list[Evaluator] = field(default_factory=list)
    output_fns: list[Output] = field(default_factory=list)
    hyperd_output_fns: list[Output] = field(default_factory=list)
    sige: SignificantEpochEvaluator | None = None


dependencies = {
    "SeriesCorrelation": Dep([CorrelationEvaluator, LossEvaluator], SeriesCorrelationOutput),
    "MatrixCorrelation": Dep([MatrixCorrelationEvaluator], MatrixCorrelationOutput),
    "RatioOverEpochs": Dep([RatioTestEvaluator], RatioOverEpochsOutput),
    "AllModels33OverEpochs": Dep([RatioTestEvaluator], AllModels33OverEpochsOutput),
    "SCurve": Dep([RatioTestEvaluator], SCurveOutput),
    "AllModelsSCurve": Dep([RatioTestEvaluator], AllModelsSCurveOutput),
    "K95Bars": Dep([K95Evaluator], None),
    "K95OverEpochs": Dep([K95Evaluator], None),
    "WithinVsBetweenCorrelation": Dep([MatrixCorrelationEvaluator, LossEvaluator], WithinVsBetweenCorrelationOutput),
    "K95-HLS": Dep([K95Evaluator], K95HLSOutput, hyperd=True),
    "Correlation-HLS": Dep([CorrelationEvaluator], CorrelationHLSOutput, hyperd=True),
    "Epochs-HLSwK95Heatmap": Dep([K95Evaluator], EpochsHLSwK95HeatmapOutput, hyperd=True),
    "SigE-HLS": Dep([CorrelationEvaluator], SigEHLSOutput, hyperd=True),
}


def get_dependencies(o_cfg):
    """
    Given an output config file, returns the list of necessary evaluators and output functions, followed by sige.
    sige will be SignificantEpochEvaluator() if any output requires it, otherwise None.
    """
    d_obj = DependenciesObject()

    evaluator_map: dict[type[Evaluator], Evaluator] = {}
    output_map: dict[type[Output], Output] = {}

    for key, subcfg in o_cfg.items():
        if subcfg.get("present", False):
            dep = dependencies.get(key)
            if dep is None:
                continue

            for ev_cls in dep.evaluators:
                if ev_cls not in evaluator_map:
                    evaluator_map[ev_cls] = ev_cls()
                    d_obj.evaluation_fns.append(evaluator_map[ev_cls])

            if dep.output_function is not None:
                out_cls = dep.output_function
                if out_cls not in output_map:
                    output_map[out_cls] = out_cls()
                    out_obj = output_map[out_cls]
                    if dep.hyperd:
                        d_obj.hyperd_output_fns.append(out_obj)
                    else:
                        d_obj.output_fns.append(out_obj)
                else:
                    out_obj = output_map[out_cls]

                d_obj.cfgs[out_obj] = subcfg

        if d_obj.sige is None and subcfg.get("epochs", False) == "sig":
            d_obj.sige = SignificantEpochEvaluator()

    return d_obj