from math import e
from Eval.PCA import K95Evaluator
from Eval.Pearson import CorrelationEvaluator, MatrixCorrelationEvaluator
from Eval.RatioExemplar import RatioTestEvaluator
from Eval.SigE import SignificantEpochEvaluator

dependencies = {
    "SeriesCorrelation": CorrelationEvaluator(),
    "MatrixCorrelation": MatrixCorrelationEvaluator(),
    "RatioOverEpochs": RatioTestEvaluator(),
    "SCurve": RatioTestEvaluator(),
    "K95Bars": K95Evaluator(),
    "K95OverEpochs": K95Evaluator()
    }


def get_dependencies(o_cfg):
    """ 
    Given an output config file, returns the list of necessary evaluators to produce the output, followed by sige.
    sige will be SignificantEpochEvaluator() if any output requires it, otherwise None.
    """
    evals = []
    sige = None

    for key, subcfg in o_cfg.items():
        if subcfg.get("present", False):
            ev = dependencies.get(key)
            if ev is not None and ev not in evals:
                evals.append(ev)
        if sige is None and subcfg.get("epochs", {}).get("sig", False):
            sige = SignificantEpochEvaluator()

    return evals, sige