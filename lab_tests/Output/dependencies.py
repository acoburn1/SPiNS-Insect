from math import e
from Eval.PCA import K95Evaluator
from Eval.Pearson import CorrelationEvaluator
from Eval.RatioExemplar import RatioTestEvaluator
from Eval.SigE import SignificantEpochEvaluator

dependencies = {
    "SeriesCorrelation": CorrelationEvaluator(),
    "MatrixCorrelation": CorrelationEvaluator(),
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
    for key in o_cfg.keys():
        if key.present == True:
            if dependencies[key] not in evals:
                evals.append(dependencies[key])
        if sige == None and key.value["epochs"]["sig"] == True:
            sige = SignificantEpochEvaluator
    return evals, sige