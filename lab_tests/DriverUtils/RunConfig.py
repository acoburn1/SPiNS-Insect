from dataclasses import dataclass
from configs.utils import resolve_cfgs
import numpy as np
import DataHelper.utils as DataUtils
from DataHelper.Probe import build_probe
import time
import os
from Model.StandardModel import StandardModel
import DriverUtils.Visual as Visual
from torch import kl_div, nn
from DataHelper import SpecialDataLoader as SDL
from DriverUtils.Zarr import build_zarr_from_results
from DriverUtils.RMutils import get_reference_matrices_m_l
from Output.PlotOutput import save_output, save_s_curve_output
from Eval.Pearson import MatrixCorrelationEpochEvaluator, SeriesCorrelationEvaluator
from Eval.PCA import K95BarsEpochEvaluator, K95OverEpochsEvaluator, build_pca_scatter
from Eval.RatioExemplar import SCurveEpochEvaluator


EVALUATORS = [
    CorrelationEvaluator(),
    K95Evaluator,
    RatioTestEvaluator()
    ]

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
        self.num_epochs = m_cfg['num_epochs']
        self.num_models = m_cfg['num_models']
        self.include_e0 = m_cfg['include_e0']

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

    def train(self):
        """ Trains models for each combination of hidden layer size and learning rate, evaluates them on the probe, and saves the activations to zarr files """
        if self.visual:
            run_t0 = time.time()

        for HLS in self.hidden_layer_range:
            for LR in self.learning_rate_range:

                activations_dir = self._add_suffix(self.activations_dir, HLS, LR)
                results = []

                if self.visual:
                    vis = Visual.VisualInfo(hls=HLS, lr=LR, model_n=self.num_models, epoch_n=self.num_epochs)
                    vis.start_pair()
                    for i in range(self.num_models):
                        vis.model_i = i
                        model = StandardModel(num_features=self.num_features, hidden_layer_size=HLS, batch_size=self.num_total_trials, num_epochs=self.num_epochs, learning_rate=LR, loss_fn=nn.BCEWithLogitsLoss())
                        result = model.train_eval(self.dataloader, self.X_probe, include_e0=self.include_e0, vis=vis)
                        results.append(result)
                    Visual.progress_done(vis)
                    time.sleep(.1)
                    Visual.print_dim(f"Saving zarr to {activations_dir}...")
                else:
                    for i in range(self.num_models):
                        model = StandardModel(num_features=self.num_features, hidden_layer_size=HLS, batch_size=self.num_total_trials, num_epochs=self.num_epochs, learning_rate=LR, loss_fn=nn.BCEWithLogitsLoss())
                        result = model.train_eval(self.dataloader, self.X_probe, include_e0=self.include_e0)
                        results.append(result)

                build_zarr_from_results(f"{activations_dir}/activations.zarr", results)
        return

    def evaluate(self):
        """ Evaluates the trained models on the probe using the specified evaluators and saves the results to npz files """
        for HLS in self.hidden_layer_range:
            for LR in self.learning_rate_range:

    def generate_output(self):
        pass

    @staticmethod
    def _add_suffix(d: str, hls: int, lr: float) -> str:
        lr_str = f"{lr}".replace(".", "p")
        return d + f"_hls{hls}_lr{lr_str}"
