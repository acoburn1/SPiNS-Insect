from dataclasses import dataclass
from Eval.Correlation import LossEvaluator
from configs.utils import resolve_cfgs
import numpy as np
import DataHelper.utils as DataUtils
from DataHelper.Probe import build_probe
import time
import os
from pathlib import Path
from Model.StandardModel import StandardModel
import DriverUtils.Visual as Visual
from torch import kl_div, nn
from DataHelper import SpecialDataLoader as SDL
from DriverUtils.Zarr import build_zarr_from_results
from DriverUtils.RMutils import get_reference_matrices_m_l
from DriverUtils.Organize import group_graphs_by_name
from Output.schema.PlotOutput import plot_output
from Output.schema.dependencies import get_dependencies
from Statistics.StatHelper import stats_over_models

@dataclass
class RunConfig:
    def __init__(self, args):
        self.visual = args.visual

        d_cfg, m_cfg, o_cfg, p_cfg, dir_cfg = resolve_cfgs(args, idv=True)
        assert not (d_cfg['num_total_trials'] / d_cfg['num_mod_trials'] == 2 and d_cfg['special_dl'] or d_cfg['num_total_trials'] / d_cfg['num_mod_trials'] != 2 and not d_cfg['special_dl']), "special data loader must be used when num mod and num lat trials ineq"

        if (self.visual):
            Visual.print_cfgs(resolve_cfgs(args))

        self.alt = d_cfg['alt']
        self.training_name = d_cfg['training_name']
        self.special_dl = d_cfg['special_dl']
        self.num_mod_trials = d_cfg['num_mod_trials']
        self.num_total_trials = d_cfg['num_total_trials']
        self.generate_rms = d_cfg['generate_rms']

        self.num_features = m_cfg['num_features']
        self.hidden_layer_range = np.arange(m_cfg['hidden_layer_range']['start'], m_cfg['hidden_layer_range']['end'] + m_cfg['hidden_layer_range']['step'], m_cfg['hidden_layer_range']['step'])
        self.learning_rate_range = np.linspace(m_cfg['learning_rate']['start'], m_cfg['learning_rate']['end'], m_cfg['learning_rate']['num'])
        self.training_epochs = m_cfg['num_epochs']
        self.eval_epochs = self.training_epochs + 1
        self.num_models = m_cfg['num_models']

        self.o_cfg = o_cfg

        self.p_cfg = p_cfg
        self.X_probe, self.probe_metadata, self.probe_index = build_probe(self.p_cfg)

        self.training_data_filename = f"{dir_cfg['training_data']}/{self.training_name}.csv"
        self.training_inputs, self.training_outputs = DataUtils.csv_training_data_to_numpy(self.training_data_filename, num_features=self.num_features)
        assert len(self.training_inputs) == self.num_total_trials, f"expected {self.num_total_trials} training trials but got {len(self.training_inputs)}. check data config"
        self.dataloader = DataUtils.get_dataloader(self.training_inputs, self.training_outputs) if not self.special_dl else SDL.SpecialDataLoader(self.training_inputs, self.training_outputs, self.num_mod_trials)

        self.modular_p_m_filename = f"{dir_cfg['reference_matrices']}/cooc-jaccard-mod.csv"
        self.lattice_p_m_filename = f"{dir_cfg['reference_matrices']}/cooc-jaccard-lat.csv" if not self.alt else "Data/ReferenceMatrices/cooc-jaccard-lat-alt.csv"
        self.mod_rm, self.lat_rm = get_reference_matrices_m_l(self.modular_p_m_filename, self.lattice_p_m_filename, self.num_mod_trials, self.training_inputs, self.generate_rms)
        self.activations_dir = f"{dir_cfg['activation_data']}/{self.training_name}"
        self.analysis_dir = f"{dir_cfg['analysis_data']}/{self.training_name}"
        self.output_dir = f"{dir_cfg['output']}/{self.training_name}"

        self.dependencies = get_dependencies(self.o_cfg)

    def train(self):
        """
        Trains models for each combination of hidden layer size and learning rate,
        evaluates them on the probe, and saves the activations to zarr files.
        """
        pair_n = int(len(self.hidden_layer_range) * len(self.learning_rate_range))
        total_models = int(pair_n * self.num_models)

        vis = None
        if self.visual:
            vis = Visual.ModelVisualInfo(
                hls=int(self.hidden_layer_range[0]),
                lr=float(self.learning_rate_range[0]),
                model_n=total_models,
                epoch_n=int(self.training_epochs),
            )
            vis.start_pair()

        global_model_i = 0
        pair_i = 0

        try:
            for HLS in self.hidden_layer_range:
                for LR in self.learning_rate_range:
                    activations_dir = self._add_suffix(self.activations_dir, HLS, LR)
                    results = []

                    if vis is not None:
                        vis.hls = int(HLS)
                        vis.lr = float(LR)

                    for i in range(self.num_models):
                        if vis is not None:
                            vis.model_i = global_model_i

                        model = StandardModel(
                            num_features=self.num_features,
                            hidden_layer_size=HLS,
                            batch_size=self.num_total_trials,
                            num_epochs=self.training_epochs,
                            learning_rate=LR,
                            loss_fn=nn.BCEWithLogitsLoss(),
                        )

                        result = model.train_eval(
                            self.dataloader,
                            self.X_probe,
                            vis=vis,  
                        )
                        results.append(result)

                        global_model_i += 1
                        
                    vis.note("saving zarr")
                    build_zarr_from_results(f"{activations_dir}/activations.zarr", results)

                    pair_i += 1

        finally:
            if vis is not None:
                vis.progress_done()

    def evaluate(self):
        """
        Evaluates activations and saves the results to npz files.
        sig_epoch data is organized as a binary mask of shape (M, E) indicating whether each epoch is significant or not.
        The rest of the data is saved as a dictionary with keys 'raw', 'mean', 'std', 'se', 'ci_lo', 'ci_hi', and 'n', where
        'raw' is always organized as one value/object per condition(s), per epoch, per model with shape (M, E, C1, ..., Cn, D) and 
        the rest of the keys are arrays of shape (E, C1, ..., Cn, D) with statistics computed across models
        """
        evaluators = self.dependencies.evaluation_fns
        sige = self.dependencies.sige

        pair_n = int(len(self.hidden_layer_range) * len(self.learning_rate_range))
        eval_n = int(len(evaluators) + (1 if sige else 0))

        vis = None
        if self.visual:
            vis = Visual.EvalVisualInfo(
                hls=int(self.hidden_layer_range[0]),
                lr=float(self.learning_rate_range[0]),
                pair_i=0,
                pair_n=pair_n,
                eval_n=eval_n,
                model_n=int(self.num_models),
                epoch_n=int(self.eval_epochs),
            )
            vis.start()

        pair_i = 0
        try:
            for HLS in self.hidden_layer_range:
                for LR in self.learning_rate_range:
                    activations_dir = self._add_suffix(self.activations_dir, HLS, LR)
                    zarr_path = f"{activations_dir}/activations.zarr"

                    analysis_dir = self._add_suffix(self.analysis_dir, HLS, LR)
                    os.makedirs(analysis_dir, exist_ok=True)

                    if vis is not None:
                        vis.hls = int(HLS)
                        vis.lr = float(LR)
                        vis.pair_i = int(pair_i)

                    for ev_i, evaluator in enumerate(evaluators):
                        if vis is not None:
                            vis.set_eval(evaluator.name, ev_i)

                        raw = evaluator.run(self, zarr_path, vis=vis)

                        stats = stats_over_models(raw)

                        np.savez(
                            f"{analysis_dir}/{evaluator.name}.npz",
                            raw=raw,
                            mean=stats["mean"],
                            std=stats["std"],
                            se=stats["se"],
                            ci_lo=stats["ci_lo"],
                            ci_hi=stats["ci_hi"],
                            n=stats["n"],
                        )

                        if sige and evaluator.name == "Correlation":
                            if vis is not None:
                                vis.set_eval("SigEpoch", ev_i + 1)
                                vis.note("vector pass")

                            sige_results = sige.run(raw)
                            np.savez(f"{analysis_dir}/sige.npz", results=sige_results)

                            if vis is not None:
                                vis.fast_done()

                    pair_i += 1

        finally:
            if vis is not None:
                vis.close()

    def generate_output(self):
        print(Visual.get_dim(f"Saving plots to {self.output_dir} (or subdirectories)..."))

        hyperd_out_fns = self.dependencies.hyperd_output_fns
        out_fns = self.dependencies.output_fns
        cfgs = self.dependencies.cfgs

        for fn in hyperd_out_fns:
            os.makedirs(self.output_dir, exist_ok=True)
            specs = fn.generate_output(cfgs[fn], self.analysis_dir)
            for spec in specs:
                plot_output(spec, f"{self.output_dir}/{fn.name}")

        for HLS in self.hidden_layer_range:
            for LR in self.learning_rate_range:
                output_dir = self._add_suffix(self.output_dir, HLS, LR)
                analysis_dir = self._add_suffix(self.analysis_dir, HLS, LR)
                for fn in out_fns:
                    os.makedirs(output_dir, exist_ok=True)
                    specs = fn.generate_output(cfgs[fn], analysis_dir)
                    for spec in specs:
                        plot_output(spec, f"{output_dir}/{fn.name}")
        
        group_graphs_by_name(Path(self.output_dir).parent)

    @staticmethod
    def _add_suffix(d: str, hls: int, lr: float) -> str:
        lr_str = f"{lr}".replace(".", "p")
        return d + f"_hls{hls}_lr{lr_str}"
