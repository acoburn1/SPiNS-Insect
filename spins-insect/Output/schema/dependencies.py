from dataclasses import dataclass, field

from Eval.Protocol import Evaluator
from Output.Protocol import Output
from Eval.PCA import K95Evaluator
from Eval.Correlation import CorrelationEvaluator, MatrixCorrelationEvaluator, LossEvaluator, WithinVsBetweenCorrelationEvaluator
from Eval.RatioExemplar import RatioTestEvaluator
from Eval.MFA import MFA
from Eval.SigE import SignificantEpochEvaluator, WithinVsBetweenSignificantEpochEvaluator
from Output.SeriesCorrelation import SeriesCorrelationOutput
from Output.MatrixCorrelation import MatrixCorrelationOutput
from Output.K95HLS import K95HLSOutput
from Output.CorrelationHLS import CorrelationHLSOutput
from Output.EpochsHLSwK95Heatmap import EpochsHLSwK95HeatmapOutput
from Output.SigEHLS import SigEHLSOutput
from Output.RatioOverEpochs import RatioOverEpochsOutput
from Output.SCurve import SCurveOutput
from Output.AllModels33OverEpochs import AllModels33OverEpochsOutput
from Output.AllModelsSCurve import AllModelsSCurveOutput
from Output.GeneralizationCorrelationDiff import GeneralizationCorrelationDiffOutput
from Output.K95Correlation import K95CorrelationOutput
from Output.K95DiffCorrelationDiff import K95DiffCorrelationDiffOutput
from Output.K95DiffGeneralization import K95DiffGeneralizationOutput
from Output.SeriesK95 import SeriesK95Output
from Output.SeriesMFA import SeriesMFA

@dataclass
class Dep:
    evaluators: list[type[Evaluator]]
    output_function: type[Output] | None

@dataclass
class DependenciesObject:
    cfgs: dict[Output, dict] = field(default_factory=dict)
    evaluation_fns: list[Evaluator] = field(default_factory=list)
    output_fns: list[Output] = field(default_factory=list)
    hyperd_output_fns: list[Output] = field(default_factory=list)
    sige: SignificantEpochEvaluator | None = None
    wb_sige: WithinVsBetweenSignificantEpochEvaluator | None = None


dependencies = {
    "SeriesCorrelation": Dep([CorrelationEvaluator, LossEvaluator], SeriesCorrelationOutput),
    "MatrixCorrelation": Dep([MatrixCorrelationEvaluator], MatrixCorrelationOutput),
    "RatioOverEpochs": Dep([RatioTestEvaluator], RatioOverEpochsOutput),
    "AllModels33OverEpochs": Dep([RatioTestEvaluator], AllModels33OverEpochsOutput),
    "SCurve": Dep([RatioTestEvaluator], SCurveOutput),
    "AllModelsSCurve": Dep([RatioTestEvaluator], AllModelsSCurveOutput),
    "K95-HLS": Dep([K95Evaluator], K95HLSOutput),
    "Correlation-HLS": Dep([CorrelationEvaluator], CorrelationHLSOutput),
    "Epochs-HLSwK95Heatmap": Dep([K95Evaluator], EpochsHLSwK95HeatmapOutput),
    "SigE-HLS": Dep([CorrelationEvaluator], SigEHLSOutput),
    "GeneralizationCorrelationDiff": Dep([RatioTestEvaluator, CorrelationEvaluator], GeneralizationCorrelationDiffOutput),
    "K95Correlation": Dep([K95Evaluator, CorrelationEvaluator], K95CorrelationOutput),
    "K95DiffCorrelationDiff": Dep([K95Evaluator, CorrelationEvaluator], K95DiffCorrelationDiffOutput),
    "K95DiffGeneralization": Dep([K95Evaluator, RatioTestEvaluator], K95DiffGeneralizationOutput),
    "SeriesK95": Dep([K95Evaluator], SeriesK95Output),
    "SeriesMFA": Dep([MFA], SeriesMFA),
}


def get_dependencies(o_cfg):
    """
    Given an output config file, returns the list of necessary evaluators and output functions, followed by sig-evaluators.
    sige is used for "sig" epoch mode and wb_sige is used for "wb-sig" epoch mode.
    """
    d_obj = DependenciesObject()

    unknown = sorted(set(o_cfg) - set(dependencies))
    if unknown:
        raise ValueError(f"Unknown output configuration key(s): {unknown}")

    evaluator_map: dict[type[Evaluator], Evaluator] = {}
    output_map: dict[type[Output], Output] = {}

    for key, subcfg in o_cfg.items():
        if subcfg.get("present", False):
            dep = dependencies[key]

            for ev_cls in dep.evaluators:
                if ev_cls not in evaluator_map:
                    evaluator_map[ev_cls] = ev_cls()
                    d_obj.evaluation_fns.append(evaluator_map[ev_cls])

            if dep.output_function is not None:
                out_cls = dep.output_function
                if out_cls not in output_map:
                    output_map[out_cls] = out_cls()
                    out_obj = output_map[out_cls]
                    if dep.output_function.hyperd:
                        d_obj.hyperd_output_fns.append(out_obj)
                    else:
                        d_obj.output_fns.append(out_obj)
                else:
                    out_obj = output_map[out_cls]

                d_obj.cfgs[out_obj] = subcfg

        epoch_mode = str(subcfg.get("epochs", "")).lower()
        sige_type = str(subcfg.get("sige_type", "")).lower()
        corr_type = str(subcfg.get("corr_type", "")).lower()
        modes = {m for m in (epoch_mode, sige_type) if m}

        if d_obj.sige is None and "sig" in modes:
            d_obj.sige = SignificantEpochEvaluator()

        if d_obj.wb_sige is None and "wb-sig" in modes:
            d_obj.wb_sige = WithinVsBetweenSignificantEpochEvaluator()
            if WithinVsBetweenCorrelationEvaluator not in evaluator_map:
                evaluator_map[WithinVsBetweenCorrelationEvaluator] = WithinVsBetweenCorrelationEvaluator()
                d_obj.evaluation_fns.append(evaluator_map[WithinVsBetweenCorrelationEvaluator])

        if corr_type == "wb" and WithinVsBetweenCorrelationEvaluator not in evaluator_map:
            evaluator_map[WithinVsBetweenCorrelationEvaluator] = WithinVsBetweenCorrelationEvaluator()
            d_obj.evaluation_fns.append(evaluator_map[WithinVsBetweenCorrelationEvaluator])

    return d_obj
